import os
import progressbar
import numpy as np
import json
import matplotlib.patches as mpatches

import scipy.optimize

import logs
import utility
import models

from testbed import lille, trento_a, trento_b

from logs import gen_estimations_from_testbed_run, gen_measurements_from_testbed_run, \
    gen_delay_estimates_from_testbed_run
import base
from base import get_dist, pair_index, convert_ts_to_sec, convert_sec_to_ts, convert_ts_to_m, convert_m_to_ts, ci_to_rd

import matplotlib
import matplotlib.pyplot as plt
from utility import slugify, cached_legacy, init_cache, load_env_config, set_global_cache_prefix_by_config
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from export import add_df_cols, load_plot_defaults, save_and_crop, c_in_air, CONFIDENCE_FILL_COLOR, PERCENTILES_FILL_COLOR, COLOR_MAP, PROTOCOL_NAME
import pandas as pd


def export_tof_simulation_response_std(config, export_dir):

    from sim_tdoa import sim

    limit = 6.0
    step = 0.1
    response_delay_exps = np.arange(-limit, limit+step, step)

    xs = response_delay_exps
    num_sims = 1000
    num_repetitions = 1
    resp_delay_s = 1.0

    rx_noise_std = 1.0e-09


    def proc():
        data_rows = []
        for x in xs:

            res, _ = sim(
                num_exchanges=num_sims,
                resp_delay_s=(resp_delay_s, resp_delay_s*np.power(10, x)),
                node_drift_std=100.0/1000000.0,
                rx_noise_map=rx_noise_std,
                tx_delay_mean=0.0,
                tx_delay_std=0.0, rx_delay_mean=0.0, rx_delay_std=0.0
            )

            data_rows.append(
                {
                    'rdr': x,
                    'tof_mean': 1.0e09*res['est_tof_a'].mean(),
                    'tof_std': 1.0e09*res['est_tof_a'].std(),
                }
            )

        return data_rows

    data_rows = cached_legacy(('export_tdoa_simulation_response_mean', limit, step, 10, num_sims, resp_delay_s), proc)

    df = pd.DataFrame(data_rows)

    df = df.rename(columns={"tof_mean": "ToF Mean", "tof_std": "ToF SD"})

    plt.clf()
    ax = df.plot.line(x='rdr', y=['ToF Mean'])

    #ax.xaxis.set_major_formatter(lambda x, pos: r'$10^{{{}}}$'.format(int(round(x))))

    #plt.ylim(0.2, 1.8)

    ax.set_axisbelow(True)
    ax.set_xlabel("Delay Ratio $\\frac{D_A}{D_B}$")
    ax.set_ylabel("Sample SD [ns]")

    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

    #ax.yaxis.set_major_locator(MultipleLocator(1.0))
    #ax.yaxis.set_minor_locator(MultipleLocator(0.2))


    # counter = 0
    # for p in ax.patches:
    #     height = p.get_height()
    #     if np.isnan(height):
    #         height = 0
    #
    #     ax.text(p.get_x() + p.get_width() / 2., height,
    #             "{:.2f}".format(height), fontsize=9, color='black', ha='center',
    #             va='bottom')
    #
    #     # ax.text(p.get_x() + p.get_width()/2., 0.5, '%.2f' % stds[offset], fontsize=12, color='black', ha='center', va='bottom')
    #     counter += 1


    plt.grid(color='lightgray', linestyle='dashed')

    plt.legend(ncol=2)
    plt.gcf().set_size_inches(6.0, 4.5)

    ticks = list(ax.get_yticks())
    labels = list(ax.get_yticklabels())

    #ticks.append(np.sqrt(0.5))
    #ticks.append(np.sqrt(2.5))

    #labels.append(r'$\sqrt{0.5}\sigma$')
    #labels.append(r'$\sqrt{2.5}\sigma$')

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)



    #print(ticks)
    #print(labels)


    plt.tight_layout()

    plt.savefig("{}/sim_rmse_reponse_delay_ratio.pdf".format(export_dir), bbox_inches = 'tight', pad_inches = 0)
    #plt.show()

    plt.close()



def export_tdoa_simulation_response_std_scatter(config, export_dir):

    from sim_tdoa import sim

    response_delay_exps = np.arange(-6.0, 6+1, 0.25)

    xs = response_delay_exps
    num_sims = 100
    resp_delay_s = 1.0


    def proc():
        data_rows = []
        for x in xs:

            _, run_data_rows = sim(
                num_exchanges=num_sims,
                resp_delay_s=(resp_delay_s, resp_delay_s * np.power(10, x)),
                node_drift_std=10.0 / 1000000.0,
                rx_noise=1.0e-09,
                tx_delay_mean=0.0,
                tx_delay_std=0.0, rx_delay_mean=0.0, rx_delay_std=0.0
            )

            for row in run_data_rows:
                data_rows.append(
                    {
                        'rdr': x,
                        'tof_std': 1.0e09*(row['est_tof_a']-row['real_tof']),
                        'tdoa_std': 1.0e09*(row['est_tdoa']-row['real_tdoa']),
                    }
                )

        return data_rows

    data_rows = cached_legacy(('export_tdoa_simulation_response_std_scatter', hash(json.dumps(list(xs))), 9, num_sims, resp_delay_s, num_sims), proc)
    df = pd.DataFrame(data_rows)
    print(df)

    df = df.rename(columns={"tof_std": "ToF SD", "tdoa_std": "TDoA"})

    plt.clf()
    #ax = df.plot.line(x='rdr', y=['ToF SD', 'TDoA'])
    ax = df.plot.scatter(x='rdr', y='ToF SD', alpha=0.2)
    #ax = df.plot.scatter(x='rdr', y='TDoA')


    plt.ylim(-4.0, 4.0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Response Ratio")
    ax.set_ylabel("Sample SD [ns]")

    # counter = 0
    # for p in ax.patches:
    #     height = p.get_height()
    #     if np.isnan(height):
    #         height = 0
    #
    #     ax.text(p.get_x() + p.get_width() / 2., height,
    #             "{:.2f}".format(height), fontsize=9, color='black', ha='center',
    #             va='bottom')
    #
    #     # ax.text(p.get_x() + p.get_width()/2., 0.5, '%.2f' % stds[offset], fontsize=12, color='black', ha='center', va='bottom')
    #     counter += 1


    plt.grid(color='lightgray', linestyle='dashed')

    plt.gcf().set_size_inches(6.0, 5.5)
    plt.tight_layout()

    plt.savefig("{}/export_tdoa_simulation_response_std_scatter.pdf".format(export_dir), bbox_inches = 'tight', pad_inches = 0)
    #plt.show()

    plt.close()


def export_tdoa_simulation_response_std(export_dir):

    from sim_tdoa import sim

    ratio_limit = 0.001
    num = 50

    num_sim_per_rep = 2000
    node_drift_std = 10.0/1000000.0
    mitigate_drift = True
    rx_noise_std = 1.0e-09
    drift_rate_std= 8.0e-08


    rx_noise_map = {
        'a-b': (10.0e-09, rx_noise_std),
        'b-a': (10.0e-09, rx_noise_std),
        'a-p': (0.0e-09, rx_noise_std),
        'b-p': (0.0e-09, rx_noise_std),
    }

    def calc_predicted_tof_bias_mean(delay_b, delay_a):
        a_b_mean = models.rx_noise_map_mean(rx_noise_map, 'a', 'b')
        b_a_mean = models.rx_noise_map_mean(rx_noise_map, 'b', 'a')

        return models.calc_predicted_tof_bias_mean(delay_b, delay_a, a_b_mean, b_a_mean)

    def calc_predicted_tdoa_bias_mean(delay_b, delay_a):
        a_b_mean = models.rx_noise_map_mean(rx_noise_map, 'a', 'b')
        b_a_mean = models.rx_noise_map_mean(rx_noise_map, 'b', 'a')
        a_p_mean = models.rx_noise_map_mean(rx_noise_map, 'a', 'p')
        b_p_mean = models.rx_noise_map_mean(rx_noise_map, 'b', 'p')

        return models.calc_predicted_tdoa_bias_mean(delay_b, delay_a, a_b_mean, b_a_mean, a_p_mean, b_p_mean)

    def calc_predicted_tof_std(delay_b, delay_a):
        a_b_std = models.rx_noise_map_std(rx_noise_map, 'a', 'b')
        b_a_std = models.rx_noise_map_std(rx_noise_map, 'b', 'a')

        return models.calc_predicted_tof_std(delay_b, delay_a, a_b_std, b_a_std)

    def calc_predicted_tdoa_std(delay_b, delay_a):
        a_b_std = models.rx_noise_map_std(rx_noise_map, 'a', 'b')
        b_a_std = models.rx_noise_map_std(rx_noise_map, 'b', 'a')
        a_p_std = models.rx_noise_map_std(rx_noise_map, 'a', 'p')
        b_p_std = models.rx_noise_map_std(rx_noise_map, 'b', 'p')

        return models.calc_predicted_tdoa_std(delay_b, delay_a, a_b_std, b_a_std, a_p_std, b_p_std)


    #@utility.cached
    def proc_simulation_response_std(
            ratio_limit = 0.001,
            num = 50,
            duration_s=0.001,
            num_sim_per_rep=1,
            num_reps=1,
            node_drift_std=0.0,
            rx_noise_map=0.0,
            mitigate_drift=True,
            drift_rate_std=0.0
    ):
        data_rows = []
        prediction_rows = []

        xs = np.linspace(ratio_limit, 1.0 - ratio_limit, num)

        for x in xs:
            delay_b = x * duration_s
            delay_a = duration_s-delay_b

            for i in range(num_reps):
                res, _ = sim(
                    num_exchanges=num_sim_per_rep,
                    resp_delay_s=(delay_b, delay_a),
                    node_drift_std=node_drift_std,
                    rx_noise_map=rx_noise_map,
                    tx_delay_mean=0.0,
                    tx_delay_std=0.0, rx_delay_mean=0.0, rx_delay_std=0.0,
                    mitigate_drift=mitigate_drift,
                    drift_rate_std=drift_rate_std
                )

                tof_std = 1.0e09 * (res['est_tof_a']).std()
                tdoa_std = 1.0e09 * (res['est_tdoa']).std()


                tof_bias_mean_ci = base.calc_ci_offsets_of_mean(tof_std, num_sim_per_rep, confidence=0.99)
                tdoa_bias_mean_ci = base.calc_ci_offsets_of_mean(tdoa_std, num_sim_per_rep, confidence=0.99)

                tof_std_ci = base.calc_ci_of_sd(tof_std, num_sim_per_rep, alpha=0.01)
                tdoa_std_ci = base.calc_ci_of_sd(tdoa_std, num_sim_per_rep, alpha=0.01)

                data_rows.append(
                    {
                        'rdr': x,
                        'tof_std': tof_std,
                        'tof_std_ci_low': tof_std_ci[0],
                        'tof_std_ci_up': tof_std_ci[1],
                        'tof_ds_std': 1.0e09 * (res['est_tof_a_ds']).std(),
                        'tdoa_std': tdoa_std,
                        'tdoa_std_ci_low': tdoa_std_ci[0],
                        'tdoa_std_ci_up': tdoa_std_ci[1],
                        'tdoa_ds_std': 1.0e09 * (res['est_tdoa_ds']).std(),
                        'tdoa_half_cor_std': 1.0e09 * (res['est_tdoa_half_cor']).std(),
                        'tdoa_ds_half_cor_std': 1.0e09 * (res['est_tdoa_ds_half_cor']).std(),
                        'tof_mean': 1.0e09 * (res['est_tof_a']).mean(),
                        'tof_ds_mean': 1.0e09 * (res['est_tof_a_ds']).mean(),
                        'tdoa_mean': 1.0e09 * (res['est_tdoa']).mean(),
                        'tdoa_ds_mean': 1.0e09 * (res['est_tdoa_ds']).mean(),
                        'tdoa_half_cor_mean': 1.0e09 * (res['est_tdoa_half_cor']).mean(),
                        'tdoa_ds_half_cor_mean': 1.0e09 * (res['est_tdoa_ds_half_cor']).mean(),
                        'tof_bias_mean': 1.0e09 * ((res['est_tof_a']).mean()-(res['real_tof']).mean()),
                        'tof_bias_mean_ci_low': tof_bias_mean_ci[0],
                        'tof_bias_mean_ci_up': tof_bias_mean_ci[1],

                        'tdoa_ds_bias_mean': 1.0e09 * ((res['est_tdoa_ds']).mean()-(res['real_tdoa']).mean()),
                        'tdoa_ds_bias_mean_ci_low': tdoa_bias_mean_ci[0],
                        'tdoa_ds_bias_mean_ci_up': tdoa_bias_mean_ci[1],

                    }
                )

            prediction_rows.append({
                'rdr': x,
                'predicted_tof_std': 1.0e09 * calc_predicted_tof_std(delay_b, delay_a),
                'predicted_tof_bias_mean': 1.0e09 * calc_predicted_tof_bias_mean(delay_b, delay_a),
                'predicted_tof_std_navratil': 1.0e09 * models.calc_predicted_tof_std_navratil(models.rx_noise_map_std(rx_noise_map, 'a', 'b'), models.rx_noise_map_std(rx_noise_map, 'b', 'a'), delay_b, delay_a),
                'predicted_tdoa_std': 1.0e09 * calc_predicted_tdoa_std(delay_b, delay_a),
                'predicted_tdoa_bias_mean': 1.0e09 * calc_predicted_tdoa_bias_mean(delay_b, delay_a),
            })
        return data_rows, prediction_rows

    duration_s = 0.001

    data_rows, prediction_rows = proc_simulation_response_std(
        ratio_limit=ratio_limit,
        num=num,
        duration_s=duration_s,
        num_sim_per_rep=num_sim_per_rep,
        num_reps=1,
        node_drift_std=node_drift_std,
        rx_noise_map=rx_noise_map,
        mitigate_drift=mitigate_drift,
        drift_rate_std=0.0
    )

    # plot Variance
    data_df = pd.DataFrame(data_rows)
    pred_df = pd.DataFrame(prediction_rows)

    data_df = data_df.rename(columns={
        "tof_std": "Simulated DS-TWR",
        "tof_ds_std": "DS-TWR",
        "tdoa_std": "Simulated DS-TDoA",
        "tdoa_ds_std": "DS-TDoA",
        "tdoa_half_cor_std": "DS-TDoA (w/ DC) SD",
        "tdoa_ds_half_cor_std": "DS-TDoA DS (w/ DC) SD",
    })

    pred_df = pred_df.rename(columns={
        "predicted_tof_std": "Analytical DS-TWR",
        "predicted_tof_std_navratil": "Analytical DS-TWR\n[Navrátil and Vejražka]",
        "predicted_tdoa_std": "Analytical DS-TDoA"
    })

    ax = pred_df.plot.line(x='rdr', y='Analytical DS-TDoA', alpha=1.0, color='C2', linestyle='--')
    ax = pred_df.plot.line(ax=ax, x='rdr', y='Analytical DS-TWR', alpha=1.0, color='C4', linestyle='--')
    ax = pred_df.plot.line(ax=ax, x='rdr', y='Analytical DS-TWR\n[Navrátil and Vejražka]', alpha=1.0, color='gray', linestyle= 'dotted')

    ax = data_df.plot.line(x='rdr', ax=ax, y='Simulated DS-TDoA', alpha=0.5, color='C2', linestyle='-', marker='.')
    ax = data_df.plot.line(x='rdr', ax=ax, y='Simulated DS-TWR', alpha=0.5, color='C4', linestyle='-')

    plt.fill_between(data_df['rdr'], data_df['tdoa_std_ci_low'], data_df['tdoa_std_ci_up'], color='C2', alpha=0.25)
    plt.fill_between(data_df['rdr'], data_df['tof_std_ci_low'], data_df['tof_std_ci_up'], color='C4', alpha=0.25)

    #data_df.plot.scatter(x='rdr', y='Simulated DS-TWR', ax=ax, c='C4', s=0.5, label='Simulated DS-TWR')
    #data_df.plot.scatter(x='rdr', y='Simulated DS-TDoA', ax=ax, c='C2', s=0.5, label='Simulated DS-TDoA')

    print("Mean", data_df['tof_mean'].mean(), data_df['tdoa_ds_mean'].mean())
    print("Bias Mean", data_df['tof_bias_mean'].mean(), data_df['tdoa_ds_bias_mean'].mean())

    #ax.xaxis.set_major_formatter(lambda x, pos: r'$10^{{{}}}$'.format(int(round(x))))

    # def formatter(x):
    #     delay_b, = x * duration_s
    #     delay_a = duration_s - delay_b
    #
    #     delay_a = delay_a*1000
    #     delay_b = delay_b*1000
    #     return r'${{{}}}:{{{}}}$'.format(round(delay_b), round(delay_a))

    #ax.xaxis.set_major_formatter(lambda x, pos: formatter(x))

    #plt.axhline(y=np.sqrt(0.5), color='C0', linestyle='dotted', label = "Analytical ToF SD")
    #plt.axhline(y=np.sqrt(2.5), color='C1', linestyle='dotted', label = "Analytical TDoA")


    #plt.ylim([0.0, 1.0])
    #plt.xlim([-0.5, 0.5])


    ax.set_axisbelow(True)
    ax.set_xlabel(r"Delay Ratio $\dfrac{D_B}{D_B+D_A}$")
    ax.set_ylabel("Sample Standard Deviation [ns]")


    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))

    #ax.xaxis.set_major_locator(MultipleLocator(1.0))

    # counter = 0
    # for p in ax.patches:
    #     height = p.get_height()
    #     if np.isnan(height):
    #         height = 0
    #
    #     ax.text(p.get_x() + p.get_width() / 2., height,
    #             "{:.2f}".format(height), fontsize=9, color='black', ha='center',
    #             va='bottom')
    #
    #     # ax.text(p.get_x() + p.get_width()/2., 0.5, '%.2f' % stds[offset], fontsize=12, color='black', ha='center', va='bottom')
    #     counter += 1


    plt.grid(color='lightgray', linestyle='dashed')


    #oldplt.gcf().set_size_inches(6.15, 5.0)
    plt.gcf().set_size_inches(5.9, 4.5)

    #ax.set_ylim(0.0, 1.0)
    #plt.ylim([0.0, 1.0])

    ticks = list(ax.get_yticks())
    labels = list(ax.get_yticklabels())

    #ticks.append(1.0e09 * np.sqrt((0.5*models.rx_noise_map_std(rx_noise_map, 'a', 'b'))**2 + (0.5*models.rx_noise_map_std(rx_noise_map, 'b', 'a'))**2))
    #labels.append(r'$\sqrt{0.5^2 \sigma_{BA}^2 + 0.5^2 \sigma_{AB}^2}$')

    ticks.append(np.sqrt(0.5))
    ticks.append(np.sqrt(2.5))
    ticks.append(np.sqrt(0.75))

    labels.append(r'$\sqrt{0.5}\sigma$')
    labels.append(r'$\sqrt{2.5}\sigma$')
    labels.append(r'$0.866\sigma$' "\n" r'$\approx \sqrt{0.75}'
                  r'\sigma$')

    ticks.append(np.sqrt(0.375))
    ticks.append(np.sqrt(1.875))
    labels.append(r'$\sqrt{0.375}\sigma$')
    labels.append(r'$\sqrt{1.875}\sigma$')

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

    ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.125))

    # access legend objects automatically created from data
    handles, labels = plt.gca().get_legend_handles_labels()

    # create manual symbols for legend
    patch = mpatches.Patch(color='lightgrey', label='99% CI')

    # add manual symbols to auto legend
    handles.append(patch)
    # legend on the side!ax2.legend(handles=handles, reverse=True, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(handles=handles, reverse=False, ncol=2, handletextpad=0.3)


    print(ticks)
    print(labels)


    plt.tight_layout()

    save_and_crop("{}/tdoa_sim_rmse_reponse_delay_ratio_{}_{}.pdf".format(export_dir, round(duration_s*1000), 0), bbox_inches = 'tight', pad_inches = 0, crop=True)
    #plt.show()

    plt.close()

    ###############################################################
    ############################################################### BIAS
    ###############################################################

    plt.clf()

    data_df = pd.DataFrame(data_rows)
    pred_df = pd.DataFrame(prediction_rows)

    data_df = data_df.rename(columns={
        "tof_bias_mean": "Simulated DS-TWR",
        "tdoa_ds_bias_mean": "Simulated DS-TDoA",
    })

    pred_df = pred_df.rename(columns={
        "predicted_tof_bias_mean": "Analytical DS-TWR",
        #"predicted_tof_std_navratil": "Analytical DS-TWR\n[Navrátil and Vejražka]",
        "predicted_tdoa_bias_mean": "Analytical DS-TDoA"
    })

    ax = pred_df.plot.line(x='rdr', y='Analytical DS-TDoA', alpha=1.0, color='C2', linestyle='--')
    ax = pred_df.plot.line(ax=ax, x='rdr', y='Analytical DS-TWR', alpha=1.0, color='C4', linestyle='--')
    #ax = pred_df.plot.line(ax=ax, x='rdr', y='Analytical DS-TWR\n[Navrátil and Vejražka]', alpha=1.0,
    #                       color='gray', linestyle='dotted')

    ax = data_df.plot.line(x='rdr', ax=ax, y='Simulated DS-TDoA', alpha=0.5, color='C2', linestyle='-',
                           marker='.')
    ax = data_df.plot.line(x='rdr', ax=ax, y='Simulated DS-TWR', alpha=0.5, color='C4', linestyle='-')

    plt.fill_between(data_df['rdr'], data_df['Simulated DS-TDoA'] + data_df['tdoa_ds_bias_mean_ci_low'], data_df['Simulated DS-TDoA'] +  data_df['tdoa_ds_bias_mean_ci_up'], color='C2',
                     alpha=0.25)
    plt.fill_between(data_df['rdr'], data_df['Simulated DS-TWR'] +  data_df['tof_bias_mean_ci_low'], data_df['Simulated DS-TWR'] +  data_df['tof_bias_mean_ci_up'], color='C4',
                     alpha=0.25)

    # data_df.plot.scatter(x='rdr', y='Simulated DS-TWR', ax=ax, c='C4', s=0.5, label='Simulated DS-TWR')
    # data_df.plot.scatter(x='rdr', y='Simulated DS-TDoA', ax=ax, c='C2', s=0.5, label='Simulated DS-TDoA')

    # ax.xaxis.set_major_formatter(lambda x, pos: r'$10^{{{}}}$'.format(int(round(x))))

    # def formatter(x):
    #     delay_b, = x * duration_s
    #     delay_a = duration_s - delay_b
    #
    #     delay_a = delay_a*1000
    #     delay_b = delay_b*1000
    #     return r'${{{}}}:{{{}}}$'.format(round(delay_b), round(delay_a))

    # ax.xaxis.set_major_formatter(lambda x, pos: formatter(x))

    # plt.axhline(y=np.sqrt(0.5), color='C0', linestyle='dotted', label = "Analytical ToF SD")
    # plt.axhline(y=np.sqrt(2.5), color='C1', linestyle='dotted', label = "Analytical TDoA")

    # plt.ylim([0.0, 1.0])
    # plt.xlim([-0.5, 0.5])
    #plt.ylim(0.0, 1.8)

    ax.set_axisbelow(True)
    ax.set_xlabel(r"Delay Ratio $\dfrac{D_B}{D_B+D_A}$")
    ax.set_ylabel("Sample Average Bias [ns]")

    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))

    # ax.xaxis.set_major_locator(MultipleLocator(1.0))

    # counter = 0
    # for p in ax.patches:
    #     height = p.get_height()
    #     if np.isnan(height):
    #         height = 0
    #
    #     ax.text(p.get_x() + p.get_width() / 2., height,
    #             "{:.2f}".format(height), fontsize=9, color='black', ha='center',
    #             va='bottom')
    #
    #     # ax.text(p.get_x() + p.get_width()/2., 0.5, '%.2f' % stds[offset], fontsize=12, color='black', ha='center', va='bottom')
    #     counter += 1

    plt.grid(color='lightgray', linestyle='dashed')

    plt.gcf().set_size_inches(6.15, 5.0)

    ticks = list(ax.get_yticks())
    labels = list(ax.get_yticklabels())

    # ticks.append(1.0e09 * np.sqrt((0.5*models.rx_noise_map_std(rx_noise_map, 'a', 'b'))**2 + (0.5*models.rx_noise_map_std(rx_noise_map, 'b', 'a'))**2))
    # labels.append(r'$\sqrt{0.5^2 \sigma_{BA}^2 + 0.5^2 \sigma_{AB}^2}$')

    ticks.append(np.sqrt(0.5))
    ticks.append(np.sqrt(2.5))
    ticks.append(np.sqrt(0.75))

    labels.append(r'$\sqrt{0.5}\sigma$')
    labels.append(r'$\sqrt{2.5}\sigma$')
    labels.append(r'$0.866\sigma$' "\n" r'$\approx \sqrt{0.75}'
                  r'\sigma$')

    ticks.append(np.sqrt(0.375))
    ticks.append(np.sqrt(1.875))
    labels.append(r'$\sqrt{0.375}\sigma$')
    labels.append(r'$\sqrt{1.875}\sigma$')

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

    ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.125))

    # access legend objects automatically created from data
    handles, labels = plt.gca().get_legend_handles_labels()

    # create manual symbols for legend
    patch = mpatches.Patch(color='lightgrey', label='99% CI')

    # add manual symbols to auto legend
    handles.append(patch)
    # legend on the side!ax2.legend(handles=handles, reverse=True, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(handles=handles, reverse=False, ncol=2, handletextpad=0.3)

    print(ticks)
    print(labels)

    plt.tight_layout()

    save_and_crop(
        "{}/tdoa_sim_rmse_reponse_bias_delay_ratio_{}_{}.pdf".format(export_dir, round(duration_s * 1000), 0),
        bbox_inches='tight', pad_inches=0, crop=True)
    # plt.show()
    plt.close()


def export_bias_comparison(export_dir):
    from sim_tdoa import sim

    num_sim_per_rep = 2000
    node_drift_std = 10.0 / 1000000.0

    rx_noise_std = 1.0e-09

    los_noise = (0.0e-09, 1.0e-09)

    nlos_bias = 4.0e-09
    nlos_std = 0.0e-09
    nlos_probability = 0.5

    # the nlos noise is a combination of the normal noise and a bias which is added with a certain probability

    def nlos_sampler(**kwargs):
        # we ignore those arguments since we calculate them manually below

        error = np.random.normal(loc=los_noise[0], scale=los_noise[1])

        if np.random.rand() < nlos_probability:
            error += np.random.normal(loc=nlos_bias, scale=nlos_std)

        return error

    nlos_noise = (los_noise[0]+nlos_probability*nlos_bias, np.sqrt(los_noise[1]**2 + nlos_probability*(nlos_std**2) + nlos_probability*(1-nlos_probability)* (nlos_bias**2)), nlos_sampler)

    # print(nlos_noise)
    # test_errs = [nlos_sampler() for _ in range(100000)]
    # print(np.mean(test_errs), np.std(test_errs))
    # exit()

    scenarios = {
        'LOS': {
            'a-b': los_noise,
            'b-a': los_noise,
            'a-p': los_noise,
            'b-p': los_noise,
        },
        'A': {
            'a-b': nlos_noise,
            'b-a': nlos_noise,
            'a-p': los_noise,
            'b-p': los_noise,
        },
        'B': {
            'a-b': los_noise,
            'b-a': los_noise,
            'a-p': nlos_noise,
            'b-p': los_noise,
        },
        'C': {
            'a-b': los_noise,
            'b-a': los_noise,
            'a-p': los_noise,
            'b-p': nlos_noise,
        },
        'AB': {
            'a-b': nlos_noise,
            'b-a': nlos_noise,
            'a-p': nlos_noise,
            'b-p': los_noise,
        },
        'AC': {
            'a-b': nlos_noise,
            'b-a': nlos_noise,
            'a-p': los_noise,
            'b-p': nlos_noise,
        },
        'BC': {
            'a-b': los_noise,
            'b-a': los_noise,
            'a-p': nlos_noise,
            'b-p': nlos_noise,
        },
        'ABC': {
            'a-b': nlos_noise,
            'b-a': nlos_noise,
            'a-p': nlos_noise,
            'b-p': nlos_noise,
        }
    }

    sim_res = {}
    pred_res = {}

    for (scenario, rx_noise_map) in scenarios.items():

        def calc_predicted_tof_bias_mean(delay_b, delay_a):
            a_b_mean = models.rx_noise_map_mean(rx_noise_map, 'a', 'b')
            b_a_mean = models.rx_noise_map_mean(rx_noise_map, 'b', 'a')

            return models.calc_predicted_tof_bias_mean(delay_b, delay_a, a_b_mean, b_a_mean)

        def calc_predicted_tdoa_bias_mean(delay_b, delay_a):
            a_b_mean = models.rx_noise_map_mean(rx_noise_map, 'a', 'b')
            b_a_mean = models.rx_noise_map_mean(rx_noise_map, 'b', 'a')
            a_p_mean = models.rx_noise_map_mean(rx_noise_map, 'a', 'p')
            b_p_mean = models.rx_noise_map_mean(rx_noise_map, 'b', 'p')

            return models.calc_predicted_tdoa_bias_mean(delay_b, delay_a, a_b_mean, b_a_mean, a_p_mean, b_p_mean)

        def calc_predicted_tof_std(delay_b, delay_a):
            a_b_std = models.rx_noise_map_std(rx_noise_map, 'a', 'b')
            b_a_std = models.rx_noise_map_std(rx_noise_map, 'b', 'a')

            return models.calc_predicted_tof_std(delay_b, delay_a, a_b_std, b_a_std)

        def calc_predicted_tdoa_std(delay_b, delay_a):

            a_b_std = models.rx_noise_map_std(rx_noise_map, 'a', 'b')
            b_a_std = models.rx_noise_map_std(rx_noise_map, 'b', 'a')
            a_p_std = models.rx_noise_map_std(rx_noise_map, 'a', 'p')
            b_p_std = models.rx_noise_map_std(rx_noise_map, 'b', 'p')

            return models.calc_predicted_tdoa_std(delay_b, delay_a, a_b_std, b_a_std, a_p_std, b_p_std)

        # @utility.cached
        def proc_sim(
                num_sim_per_rep=1,
                rx_noise_map=0.0,
                node_drift_std=0.0,
        ):
            data_rows = []
            prediction_rows = []

            duration_s = 0.001

            delay_b = 0.5 * duration_s
            delay_a = 0.5 * duration_s

            res, _ = sim(
                num_exchanges=num_sim_per_rep,
                resp_delay_s=(delay_b, delay_a),
                node_drift_std=node_drift_std,
                rx_noise_map=rx_noise_map,
                tx_delay_mean=0.0,
                tx_delay_std=0.0,
                rx_delay_mean=0.0,
                rx_delay_std=0.0,
                mitigate_drift=True,
                drift_rate_std=0.0
            )

            tof_std = 1.0e09 * (res['est_tof_a']).std()
            tdoa_std = 1.0e09 * (res['est_tdoa']).std()

            tof_bias_mean_ci = base.calc_ci_offsets_of_mean(tof_std, num_sim_per_rep, confidence=0.99)
            tdoa_bias_mean_ci = base.calc_ci_offsets_of_mean(tdoa_std, num_sim_per_rep, confidence=0.99)

            tof_std_ci = base.calc_ci_of_sd(tof_std, num_sim_per_rep, alpha=0.01)
            tdoa_std_ci = base.calc_ci_of_sd(tdoa_std, num_sim_per_rep, alpha=0.01)

            data_rows.append(
                {
                #                'rdr': x,
                    'tof_std': tof_std,
                #    'tof_std_ci_low': tof_std_ci[0],
                #    'tof_std_ci_up': tof_std_ci[1],
                #    'tof_ds_std': 1.0e09 * (res['est_tof_a_ds']).std(),
                #    'tdoa_std': tdoa_std,
                #    'tdoa_std_ci_low': tdoa_std_ci[0],
                #    'tdoa_std_ci_up': tdoa_std_ci[1],
                    'tdoa_ds_std': 1.0e09 * (res['est_tdoa_ds']).std(),
                #    'tdoa_half_cor_std': 1.0e09 * (res['est_tdoa_half_cor']).std(),
                #    'tdoa_ds_half_cor_std': 1.0e09 * (res['est_tdoa_ds_half_cor']).std(),
                #    'tof_mean': 1.0e09 * (res['est_tof_a']).mean(),
                #    'tof_ds_mean': 1.0e09 * (res['est_tof_a_ds']).mean(),
                #    'tdoa_mean': 1.0e09 * (res['est_tdoa']).mean(),
                #    'tdoa_ds_mean': 1.0e09 * (res['est_tdoa_ds']).mean(),
                #    'tdoa_half_cor_mean': 1.0e09 * (res['est_tdoa_half_cor']).mean(),
                #    'tdoa_ds_half_cor_mean': 1.0e09 * (res['est_tdoa_ds_half_cor']).mean(),
                    'tof_bias_mean': 1.0e09 * ((res['est_tof_a']).mean() - (res['real_tof']).mean()),
                #    'tof_bias_mean_ci_low': tof_bias_mean_ci[0],
                #    'tof_bias_mean_ci_up': tof_bias_mean_ci[1],
                    'tdoa_ds_bias_mean': 1.0e09 * ((res['est_tdoa_ds']).mean() - (res['real_tdoa']).mean()),
                 #   'tdoa_ds_bias_mean_ci_low': tdoa_bias_mean_ci[0],
                 #   'tdoa_ds_bias_mean_ci_up': tdoa_bias_mean_ci[1],

                }
            )

            prediction_rows.append({
                'predicted_tof_std': 1.0e09 * calc_predicted_tof_std(delay_b, delay_a),
                'predicted_tof_bias_mean': 1.0e09 * calc_predicted_tof_bias_mean(delay_b, delay_a),
                'predicted_tof_std_navratil': 1.0e09 * models.calc_predicted_tof_std_navratil(
                    models.rx_noise_map_std(rx_noise_map, 'a', 'b'), models.rx_noise_map_std(rx_noise_map, 'b', 'a'), delay_b, delay_a),
                'predicted_tdoa_std': 1.0e09 * calc_predicted_tdoa_std(delay_b, delay_a),
                'predicted_tdoa_bias_mean': 1.0e09 * calc_predicted_tdoa_bias_mean(delay_b, delay_a),
            })
            return data_rows, prediction_rows


        drs, pred_rows = proc_sim(
            num_sim_per_rep=num_sim_per_rep,
            node_drift_std=node_drift_std,
            rx_noise_map=rx_noise_map,
        )

        sim_res[scenario] = drs[0]
        pred_res[scenario] = pred_rows[0]
    #
    fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[1, 2], sharey=False)
    #fig.subplots_adjust(hspace=0.01)

    #fig, ax1 = plt.subplots()

    # ax.xaxis.set_major_formatter(lambda x, pos: formatter(x))
    #ax1.yaxis.set_major_formatter(lambda x, pos: np.round(x * 100.0, 1))  # scale to cm

    ax1.set_xlabel("NLOS Multipath Scenario")
    ax1.set_ylabel('Mean Error [ns]')
    ax1.grid(color='lightgray', linestyle='dashed')

    twr_sims = ['LOS', 'A']
    twr_xs = np.arange(len(twr_sims))
    twr_means = { 'Analytical DS-TWR': [pred_res[x]['predicted_tof_bias_mean'] for x in twr_sims], 'Simulated DS-TWR': [sim_res[x]['tof_bias_mean'] for x in twr_sims]}
    twr_stds = { 'Analytical DS-TWR': [pred_res[x]['predicted_tof_std'] for x in twr_sims], 'Simulated DS-TWR': [sim_res[x]['tof_std'] for x in twr_sims]}
    twr_sim_std = { x: sim_res[x]['tof_std'] for x in twr_sims}
    twr_pred_means = { }
    twr_pred_std = { x: pred_res[x]['predicted_tof_std'] for x in twr_sims}

    #tdoa_sims = ['LOS', 'A', 'B', 'AB', 'C', 'AC', 'BC', 'ABC']
    tdoa_sims = ['LOS', 'A', 'B', 'AB', 'C', 'AC', 'BC', 'ABC']
    tdoa_xs = np.arange(len(tdoa_sims))
    tdoa_means = { 'Analytical DS-TDoA': [pred_res[x]['predicted_tdoa_bias_mean'] for x in tdoa_sims], 'Simulated DS-TDoA': [sim_res[x]['tdoa_ds_bias_mean'] for x in tdoa_sims]}
    tdoa_stds = { 'Analytical DS-TDoA': [pred_res[x]['predicted_tdoa_std'] for x in tdoa_sims], 'Simulated DS-TDoA': [sim_res[x]['tdoa_ds_std'] for x in tdoa_sims]}

    width = 0.45  # the width of the bars
    multiplier = 0

    alpha = [1.0, 0.5]
    labelcolors = ['black', 'gray']
    for attribute, measurement in twr_means.items():
        offset = width * multiplier + 0.5 * width
        rects = ax1.bar(twr_xs + offset, measurement, width, label=attribute, color='C4', alpha=alpha[multiplier]) #yerr=twr_stds[multiplier], capsize=0)
        #ax1.bar_label(rects, padding=3)
        ax1.bar_label(rects, padding=2, fontsize=8, label_type='edge',
                      labels=["{:.1f}\n[{:.1f}]".format(round(measurement[i], 1), round(twr_stds[attribute][i], 1)) for
                              i in
                              range(len(measurement))], color=labelcolors[multiplier])
        multiplier += 1



    # counter = 0
    # for p in ax1.patches:
    #     height = p.get_height()
    #
    #
    #     if np.isnan(height):
    #         height = 0
    #
    #     ax1.text(p.get_x() + p.get_width() / 2., height + (twr_sim_std + twr_pred_std)[counter],
    #             "{:.1f}\n[{:.1f}]".format(height, (twr_sim_std + twr_pred_std)[counter]), fontsize=9, color='black', ha='center',
    #             va='bottom')
    #
    #     # ax.text(p.get_x() + p.get_width()/2., 0.5, '%.2f' % stds[offset], fontsize=12, color='black', ha='center', va='bottom')
    #     counter += 1
    ax1.legend(labelcolor=labelcolors)

    ax1.set_xticks(twr_xs + width, twr_sims)

    ax1.set_ylim(-0.19, 2.45)

    # fig.set_size_inches(3.0, 3.0)
    # fig.tight_layout()
    # save_and_crop("{}/simulation_bias_nlos_scenarios_twr.pdf".format(export_dir), bbox_inches='tight',
    #               crop=True)

    #plt.clf()

    #fig, ax2 = plt.subplots()
    ax2.set_ylim(-2.45, 2.45)

    #ax1.legend(reverse=True)
    # plt.tight_layout()


    ax2.set_xlabel("NLOS Multipath Scenario")
    ax2.set_ylabel('Mean Error [ns]')

    ax2.grid(color='lightgray', linestyle='dashed')

    width = 0.45  # the width of the bars
    multiplier = 0

    alpha=[1.0, 0.5]
    labelcolors=['black', 'gray']
    hatch=[None, None]
    for attribute, measurement in tdoa_means.items():
        offset = width * multiplier + 0.5 * width
        rects = ax2.bar(tdoa_xs + offset, measurement, width, label=attribute, color='C2', alpha=alpha[multiplier], hatch=hatch[multiplier])  # yerr=twr_stds[multiplier], capsize=0)
        ax2.bar_label(rects, padding=2, fontsize=8, label_type='edge',
                     labels=["{:.1f}\n[{:.1f}]".format(round(measurement[i], 1), round(tdoa_stds[attribute][i], 1)) for i in
                             range(len(measurement))], color=labelcolors[multiplier])
        # ax1.bar_label(rects, padding=3)
        multiplier += 1

    ax2.set_xticks(tdoa_xs + width, tdoa_sims)

    ax2.legend(labelcolor=labelcolors)

    fig.set_size_inches(8.5, 4.43)
    fig.tight_layout()
    save_and_crop("{}/simulation_bias_nlos_scenarios_tdoa.pdf".format(export_dir), bbox_inches='tight',
                  crop=True)

    plt.close()

if __name__ == '__main__':
    config = load_env_config()
    load_plot_defaults()
    assert 'EXPORT_DIR' in config and config['EXPORT_DIR']
    if 'CACHE_DIR' in config and config['CACHE_DIR']:
        init_cache(config['CACHE_DIR'])

    export_bias_comparison(config['EXPORT_DIR'])
    export_tdoa_simulation_response_std(config['EXPORT_DIR'])




