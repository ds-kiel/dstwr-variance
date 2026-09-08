import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import base
import models
from utility import cached_legacy, init_cache, load_env_config

from sim_tdoa import sim

# ----------------------------------------------------------------------------
# NLOS sweep (paper Todos §2 / Review A): instead of the single bimodal case
# (4 ns bias, 50% probability), sweep the NLOS bias magnitude and the
# occurrence probability and verify the analytic model against Monte-Carlo
# across the whole grid. The magnitude grid spans the empirically measured
# NLOS bias range 0.28–3.42 m ~ 0.9–11.4 ns (Zhang'26, Yogesh'24, Yang'24).
# ----------------------------------------------------------------------------

SIGMA_LOS = 1.0e-09  # LOS reception noise SD, as in the paper
NODE_DRIFT_STD = 10.0 / 1000000.0  # 10 ppm, as in the paper
DURATION_S = 0.001  # symmetric response delays as in the existing NLOS figure

BIAS_GRID_NS = [1.0, 2.0, 4.0, 8.0, 12.0]
PROB_GRID = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

NUM_SIM_VERIFICATION = 20000
NUM_SIM_FIGURE = 50000

LINKS = ['a-b', 'b-a', 'a-p', 'b-p']

# Obstacle positions as in the paper figure (fig:sim_scenario):
# A impedes the initiator<->responder path (both directions),
# B the initiator<->listener path, C the responder<->listener path.
# NOTE: export_tdoa_simulation_response_std.py's export_bias_comparison
# defines scenario 'C' identically to 'AC' (copy-paste bug); here C = {b-p}.
SCENARIO_LINKS = {
    'LOS': set(),
    'A': {'a-b', 'b-a'},
    'B': {'a-p'},
    'C': {'b-p'},
    'AB': {'a-b', 'b-a', 'a-p'},
    'AC': {'a-b', 'b-a', 'b-p'},
    'BC': {'a-p', 'b-p'},
    'ABC': {'a-b', 'b-a', 'a-p', 'b-p'},
}

SYMMETRIC_SIGMAS = {l: SIGMA_LOS for l in LINKS}
# optional per-path asymmetric LOS noise (Todos §2)
ASYMMETRIC_SIGMAS = {
    'a-b': 1.0e-09,
    'b-a': 1.2e-09,
    'a-p': 0.8e-09,
    'b-p': 1.5e-09,
}


def make_noise_entry(sigma_los, nlos_bias=0.0, nlos_probability=0.0, nlos_std=0.0):
    """(mean, std, sampler) of eps_LOS + beta*(nlos_bias + eps_NLOS), beta~Bern(p)."""

    if nlos_probability == 0.0 or nlos_bias == 0.0:
        return (0.0, sigma_los)

    def sampler(**kwargs):
        error = np.random.normal(loc=0.0, scale=sigma_los)
        if np.random.rand() < nlos_probability:
            error += np.random.normal(loc=nlos_bias, scale=nlos_std)
        return error

    p = nlos_probability
    mean = p * nlos_bias
    std = np.sqrt(sigma_los ** 2 + p * (nlos_std ** 2) + p * (1 - p) * (nlos_bias ** 2))
    return (mean, std, sampler)


def build_noise_map(scenario, sigmas, nlos_bias, nlos_probability):
    nlos_links = SCENARIO_LINKS[scenario]
    return {
        link: make_noise_entry(sigmas[link], nlos_bias if link in nlos_links else 0.0,
                               nlos_probability if link in nlos_links else 0.0)
        for link in LINKS
    }


def predict(noise_map, delay_b, delay_a):
    means = {l: models.rx_noise_map_mean(noise_map, *l.split('-')) for l in LINKS}
    stds = {l: models.rx_noise_map_std(noise_map, *l.split('-')) for l in LINKS}
    return {
        'tof_bias': models.calc_predicted_tof_bias_mean(delay_b, delay_a, means['a-b'], means['b-a']),
        'tof_std': models.calc_predicted_tof_std(delay_b, delay_a, stds['a-b'], stds['b-a']),
        'tdoa_bias': models.calc_predicted_tdoa_bias_mean(delay_b, delay_a, means['a-b'], means['b-a'], means['a-p'], means['b-p']),
        'tdoa_std': models.calc_predicted_tdoa_std(delay_b, delay_a, stds['a-b'], stds['b-a'], stds['a-p'], stds['b-p']),
    }


def run_cell(scenario, sigmas, nlos_bias_ns, nlos_probability, num_sim, seed):
    """One Monte-Carlo cell; returns simulated + predicted moments in ns."""
    delay_b = 0.5 * DURATION_S
    delay_a = 0.5 * DURATION_S

    noise_map = build_noise_map(scenario, sigmas, nlos_bias_ns * 1.0e-09, nlos_probability)
    pred = predict(noise_map, delay_b, delay_a)

    np.random.seed(seed)
    res, _ = sim(
        num_exchanges=num_sim,
        resp_delay_s=(delay_b, delay_a),
        node_drift_std=NODE_DRIFT_STD,
        rx_noise_map=noise_map,
        tx_delay_mean=0.0, tx_delay_std=0.0, rx_delay_mean=0.0, rx_delay_std=0.0,
        mitigate_drift=True,
        drift_rate_std=0.0,
    )

    row = {
        'scenario': scenario,
        'nlos_bias_ns': nlos_bias_ns,
        'nlos_probability': nlos_probability,
        'num_sim': num_sim,
    }
    for est, sim_key in (('tof', 'est_tof_a'), ('tdoa', 'est_tdoa_ds')):
        real_key = 'real_tof' if est == 'tof' else 'real_tdoa'
        err = res[sim_key] - res[real_key]
        mc_bias = 1.0e09 * err.mean()
        mc_std = 1.0e09 * err.std(ddof=1)
        bias_ci = base.calc_ci_offsets_of_mean(mc_std, num_sim, confidence=0.99, mean=mc_bias)
        std_ci = base.calc_ci_of_sd(mc_std, num_sim, alpha=0.01)
        row.update({
            f'{est}_bias_pred': 1.0e09 * pred[f'{est}_bias'],
            f'{est}_bias_sim': mc_bias,
            f'{est}_bias_ci_low': bias_ci[0],
            f'{est}_bias_ci_up': bias_ci[1],
            f'{est}_std_pred': 1.0e09 * pred[f'{est}_std'],
            f'{est}_std_sim': mc_std,
            f'{est}_std_ci_low': std_ci[0],
            f'{est}_std_ci_up': std_ci[1],
        })
    return row


def seed_for(*parts):
    # stable, python-hash-randomization-free seed per cell
    import zlib
    return zlib.crc32('|'.join(str(p) for p in parts).encode()) % (2 ** 31)


# ----------------------------------------------------------------------------
# 1) Verification sweep: all NLOS scenarios x bias grid x probability grid,
#    symmetric and asymmetric per-path LOS noise.
# ----------------------------------------------------------------------------

def run_verification(config_name, sigmas, num_sim=NUM_SIM_VERIFICATION):
    scenarios = [s for s in SCENARIO_LINKS if s != 'LOS']

    def proc():
        rows = []
        total = len(scenarios) * len(BIAS_GRID_NS) * len(PROB_GRID)
        done = 0
        t0 = time.time()
        for scenario in scenarios:
            for b in BIAS_GRID_NS:
                for p in PROB_GRID:
                    rows.append(run_cell(scenario, sigmas, b, p, num_sim,
                                         seed_for('verify', config_name, scenario, b, p, num_sim)))
                    done += 1
                    print(f'[{config_name}] {done}/{total} ({time.time() - t0:.0f}s)', flush=True)
        return rows

    rows = cached_legacy(('nlos_sweep_verification', config_name,
                          sorted(sigmas.items()), BIAS_GRID_NS, PROB_GRID, num_sim, 1), proc)
    df = pd.DataFrame(rows)
    df.insert(0, 'config', config_name)

    for est in ('tof', 'tdoa'):
        df[f'{est}_bias_dev'] = df[f'{est}_bias_sim'] - df[f'{est}_bias_pred']
        df[f'{est}_std_rel_dev'] = df[f'{est}_std_sim'] / df[f'{est}_std_pred'] - 1.0
        df[f'{est}_bias_in_ci'] = (df[f'{est}_bias_ci_low'] <= df[f'{est}_bias_pred']) & (df[f'{est}_bias_pred'] <= df[f'{est}_bias_ci_up'])
        df[f'{est}_std_in_ci'] = (df[f'{est}_std_ci_low'] <= df[f'{est}_std_pred']) & (df[f'{est}_std_pred'] <= df[f'{est}_std_ci_up'])
    return df


def print_verification_summary(df):
    for config in df['config'].unique():
        sub = df[df['config'] == config]
        print(f'--- {config}: {len(sub)} cells ---')
        for est, name in (('tof', 'DS-TWR'), ('tdoa', 'DS-TDoA')):
            print('  {:8s} max |bias dev| = {:.3f} ns, max |SD rel dev| = {:.2%}, '
                  '99% CI coverage: bias {:.1%}, SD {:.1%}'.format(
                      name,
                      sub[f'{est}_bias_dev'].abs().max(),
                      sub[f'{est}_std_rel_dev'].abs().max(),
                      sub[f'{est}_bias_in_ci'].mean(),
                      sub[f'{est}_std_in_ci'].mean()))


# ----------------------------------------------------------------------------
# 2) Paper figures: bias and SD vs. NLOS probability, one curve per bias
#    magnitude, analytic prediction overlaid on Monte-Carlo.
#    DS-TWR under scenario A, DS-TDoA under scenario B (visible bias +p*b).
# ----------------------------------------------------------------------------

def figure_sweep(export_dir, scenario, est, color_map_name, sim_label, num_sim=NUM_SIM_FIGURE):
    def proc():
        rows = []
        for b in BIAS_GRID_NS:
            for p in PROB_GRID:
                rows.append(run_cell(scenario, SYMMETRIC_SIGMAS, b, p, num_sim,
                                     seed_for('figure', scenario, b, p, num_sim)))
                print(f'[figure {scenario}] b={b} p={p}', flush=True)
        return rows

    rows = cached_legacy(('nlos_sweep_figure', scenario, est, BIAS_GRID_NS, PROB_GRID, num_sim, 1), proc)
    df = pd.DataFrame(rows)

    p_fine = np.linspace(0.0, 1.0, 201)
    delay_b = delay_a = 0.5 * DURATION_S

    cmap = matplotlib.colormaps[color_map_name]
    colors = {b: cmap(x) for b, x in zip(BIAS_GRID_NS, np.linspace(0.45, 0.95, len(BIAS_GRID_NS)))}

    fig, (ax_bias, ax_std) = plt.subplots(1, 2, sharex=True)

    for b in BIAS_GRID_NS:
        pred_bias = []
        pred_std = []
        for p in p_fine:
            noise_map = build_noise_map(scenario, SYMMETRIC_SIGMAS, b * 1.0e-09, p)
            pred = predict(noise_map, delay_b, delay_a)
            pred_bias.append(1.0e09 * pred[f'{est}_bias'])
            pred_std.append(1.0e09 * pred[f'{est}_std'])

        sub = df[df['nlos_bias_ns'] == b]
        color = colors[b]

        ax_bias.plot(p_fine, pred_bias, linestyle='--', color=color, label=f'${b:.0f}\\,ns$')
        ax_bias.errorbar(sub['nlos_probability'], sub[f'{est}_bias_sim'],
                         yerr=[sub[f'{est}_bias_sim'] - sub[f'{est}_bias_ci_low'],
                               sub[f'{est}_bias_ci_up'] - sub[f'{est}_bias_sim']],
                         fmt='o', ms=4, color=color, linestyle='none')

        ax_std.plot(p_fine, pred_std, linestyle='--', color=color)
        ax_std.errorbar(sub['nlos_probability'], sub[f'{est}_std_sim'],
                        yerr=[sub[f'{est}_std_sim'] - sub[f'{est}_std_ci_low'],
                              sub[f'{est}_std_ci_up'] - sub[f'{est}_std_sim']],
                        fmt='o', ms=4, color=color, linestyle='none')

    for ax, ylabel in ((ax_bias, 'Mean Error [ns]'), (ax_std, 'Sample SD [ns]')):
        ax.set_xlabel('NLOS Probability $p$')
        ax.set_ylabel(ylabel)
        ax.grid(color='lightgray', linestyle='dashed')
        ax.set_axisbelow(True)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.125))

    # combined legend: magnitude colors + line/marker semantics
    from matplotlib.lines import Line2D
    handles, labels = ax_bias.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, matplotlib.container.ErrorbarContainer) else h for h in handles]
    handles += [Line2D([], [], color='gray', linestyle='--'),
                Line2D([], [], color='gray', marker='o', linestyle='none', ms=4)]
    labels += [f'Analytical {sim_label}', f'Simulated {sim_label}']
    ax_bias.legend(handles, labels, ncol=2, fontsize=9, handletextpad=0.3, columnspacing=0.8)

    fig.set_size_inches(8.5, 3.4)
    fig.tight_layout()

    path = os.path.join(export_dir, f'simulation_nlos_sweep_{est if est != "tof" else "twr"}.pdf')
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# pdfcrop via the texlive docker image (pdfcrop is not installed locally)
# ----------------------------------------------------------------------------

def crop_pdf(path):
    directory = os.path.dirname(os.path.abspath(path))
    name = os.path.basename(path)
    cropped = os.path.splitext(name)[0] + '_cropped.pdf'
    if shutil.which('pdfcrop'):
        subprocess.run(['pdfcrop', name, cropped], cwd=directory, check=True, stdout=subprocess.DEVNULL)
    else:
        subprocess.run(['docker', 'run', '--rm', '-v', f'{directory}:/work', '-w', '/work',
                        'texlive/texlive:latest', 'pdfcrop', name, cropped],
                       check=True, stdout=subprocess.DEVNULL)
    return os.path.join(directory, cropped)


if __name__ == '__main__':
    config = load_env_config()
    assert 'EXPORT_DIR' in config and config['EXPORT_DIR']
    export_dir = config['EXPORT_DIR']
    if 'CACHE_DIR' in config and config['CACHE_DIR']:
        init_cache(config['CACHE_DIR'])

    plt.rc('lines', linewidth=2.0)
    plt.rc('legend', framealpha=1.0, fancybox=True)
    plt.rc('errorbar', capsize=3)
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)
    plt.rc('font', size=11)
    plt.rcParams['axes.axisbelow'] = True

    df_sym = run_verification('symmetric_sigma', SYMMETRIC_SIGMAS)
    df_asym = run_verification('asymmetric_sigma', ASYMMETRIC_SIGMAS)
    df_all = pd.concat([df_sym, df_asym], ignore_index=True)
    csv_path = os.path.join(export_dir, 'nlos_sweep_verification.csv')
    df_all.to_csv(csv_path, index=False)
    print(f'wrote {csv_path}')
    print_verification_summary(df_all)

    figures = [
        figure_sweep(export_dir, 'A', 'tof', 'Purples', 'DS-TWR'),
        figure_sweep(export_dir, 'B', 'tdoa', 'Greens', 'DS-TDoA'),
    ]
    for path in figures:
        cropped = crop_pdf(path)
        print(f'wrote {path} / {cropped}')
