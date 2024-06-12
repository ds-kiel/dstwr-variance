import os
import progressbar
import numpy as np
import json

import scipy.optimize

import logs
import utility
from base import SPEED_OF_LIGHT
from testbed import lille, trento_a, trento_b

from logs import gen_estimations_from_testbed_run, gen_measurements_from_testbed_run, \
    gen_delay_estimates_from_testbed_run
from base import get_dist, pair_index, convert_ts_to_sec, convert_sec_to_ts, convert_ts_to_m, convert_m_to_ts, ci_to_rd

import matplotlib
import matplotlib.pyplot as plt
from utility import slugify, cached_legacy, init_cache, load_env_config, set_global_cache_prefix_by_config
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from export import add_df_cols, load_plot_defaults, save_and_crop, c_in_air, CONFIDENCE_FILL_COLOR, PERCENTILES_FILL_COLOR, COLOR_MAP, PROTOCOL_NAME
import pandas as pd


from export_ping_pong_std import prepare_df, get_df


def calc_predicted_tof_std(delay_a, delay_b, a_b_std, b_a_std):
    return np.sqrt(
        (0.5 * b_a_std) ** 2
        + (0.5 * (delay_b / (delay_a + delay_b)) * a_b_std) ** 2
        + (0.5 * (1 - (delay_b / (delay_a + delay_b))) * a_b_std) ** 2
    )


def calc_predicted_tdoa_std(delay_a, delay_b, a_b_std, b_a_std, a_p_std, b_p_std):
    comb_delay = delay_a + delay_b
    return np.sqrt(
        (0.5 * b_a_std) ** 2
        + (0.5 * a_b_std * (delay_b / comb_delay - 1)) ** 2
        + (0.5 * a_b_std * (delay_b / comb_delay)) ** 2
        + (a_p_std * (1 - delay_b / comb_delay)) ** 2
        + b_p_std ** 2
        + (a_p_std * (delay_b / comb_delay)) ** 2
    )

def export(export_dir):
    xs = np.logspace(-15, -9)
    delay_a = 0.5
    delay_b = 0.5
    tof_ys = [calc_predicted_tof_std(delay_a, delay_b, x, x)*SPEED_OF_LIGHT for x in xs]
    td_ys = [calc_predicted_tdoa_std(delay_a, delay_b, x, x, x, x)*SPEED_OF_LIGHT for x in xs]



    fig, ax = plt.subplots()
    # this we need actually! plt.axvline(CHOSEN_DUR*0.75, color='r', linestyle='--', label="Chosen Duration ({} ms)".format(round(CHOSEN_DUR*0.75)))

    # ax.set_ylim(0, 0.05)

   # plt.yscale('log')
    #plt.xscale('log')

    plt.plot(xs, tof_ys)
    plt.plot(xs, td_ys)

    save_and_crop("{}/variance_by_noise.pdf".format(export_dir), bbox_inches='tight')  # , pad_inches=0)
    plt.show()
    plt.close()

if __name__ == '__main__':
    config = load_env_config()
    load_plot_defaults()
    assert 'EXPORT_DIR' in config and config['EXPORT_DIR']
    if 'CACHE_DIR' in config and config['CACHE_DIR']:
        init_cache(config['CACHE_DIR'])

    export(config['EXPORT_DIR'])




