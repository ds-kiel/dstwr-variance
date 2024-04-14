import os
import progressbar
import numpy as np
import json

import scipy.optimize

import logs
import utility

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


logfiles = [
        '2024-02-28_ping_pong_200/job_11985.tar.gz',
        '2024-02-28_ping_pong_200/job_11986.tar.gz',
        #'2024-02-28_ping_pong_200/job_11987.tar.gz',
        #'2024-02-28_ping_pong_200/job_11988.tar.gz',
        # '2024-02-28_ping_pong_200/job_11989.tar.gz',
        # '2024-02-28_ping_pong_200/job_11990.tar.gz',
        # '2024-02-28_ping_pong_200/job_11991.tar.gz',
        # '2024-02-28_ping_pong_200/job_11992.tar.gz',
        # '2024-02-28_ping_pong_200/job_11993.tar.gz',
        # '2024-02-28_ping_pong_200/job_11994.tar.gz',
        # '2024-02-28_ping_pong_200/job_11995.tar.gz',
        # '2024-02-28_ping_pong_200/job_11996.tar.gz',
        # '2024-02-28_ping_pong_200/job_11997.tar.gz',
        # '2024-02-28_ping_pong_200/job_11998.tar.gz',
        # '2024-02-28_ping_pong_200/job_11999.tar.gz',
        # '2024-03-01_ping_pong_200/job_12004.tar.gz',
        # '2024-03-01_ping_pong_200/job_12005.tar.gz',
        # '2024-03-01_ping_pong_200/job_12006.tar.gz',
        # '2024-03-01_ping_pong_200/job_12007.tar.gz',
        # '2024-03-01_ping_pong_200/job_12008.tar.gz',
        # '2024-03-01_ping_pong_200/job_12009.tar.gz',
        # '2024-03-01_ping_pong_200/job_12010.tar.gz',
        # '2024-03-01_ping_pong_200/job_12011.tar.gz',
        # '2024-03-01_ping_pong_200/job_12012.tar.gz',
]

CHOSEN_DUR=200


def export_testbed_layouts(export_dir):

    # we draw every layout

    # TODO: Extract them from the actual data!
    high_variance_connections = {
        'trento_a': [],#[(6,3)],
        'trento_b': [],
        'lille': [] #[(11,3), (10,3), (7,1), (5,0)],
    }

    always_drawn ={
        'lille': [],
        'trento_a': [],
        'trento_b': []
    }

    limits = {
        'lille': [-0.5, 10.0, 29, 22.5],
        'trento_a': [71, 81, -1.25, 8.5],
        'trento_b': [118.5, 131.5, -1, 8.0]
    }

    node_colors = {
        'trento_a': ['C4', 'C1', 'C2', 'C5', 'C3', 'C7', 'C6'],
    }

    node_markers = {
        'trento_a': ["o"]*7,
    }
    marker_size = 4.0


    for (c,t) in enumerate([trento_a]):
        ys = []
        xs = []
        ns = []

        for (i,k) in enumerate(t.devs):
            pos = t.dev_positions[k]
            xs.append(pos[0])
            ys.append(pos[1])
            n = i+1 #k.replace("dwm1001.", "").replace("dwm1001-", "")
            ns.append(n)

        plt.clf()



        fig, ax = plt.subplots()

        t.draw_layout(plt)


        ax.set_aspect('equal', adjustable='box')


        for (a,b) in high_variance_connections[t.name]:
            plt.plot([xs[a], xs[b]], [ys[a], ys[b]], 'r', zorder=-10)

        for a in always_drawn[t.name]:
            for b in range(len(t.devs)):
                if a > b:
                    plt.plot([xs[a], xs[b]], [ys[a], ys[b]], 'r', zorder=-10)


        for i, txt in enumerate(ns):
            ax.scatter([xs[i]], [ys[i]], color=node_colors[t.name][i], marker=node_markers[t.name][i])
            ax.annotate(txt, (xs[i], ys[i]), xytext=(-3, 0), textcoords='offset points', va='center', ha='right')

        ax.set_axisbelow(True)
        ax.set_xlabel("Position X [m]")
        ax.set_ylabel("Position Y [m]")

        if t.name == 'lille':
            ax.invert_yaxis()

        plt.axis(limits[t.name])

        fig.set_size_inches(4.0, 3.5)
        plt.tight_layout()

        #plt.show()
        save_and_crop("{}/layout_{}.pdf".format(export_dir, t.name),bbox_inches = 'tight', pad_inches = 0, dpi=600, crop=True)

        plt.close()

if __name__ == '__main__':
    config = load_env_config()
    load_plot_defaults()
    assert 'EXPORT_DIR' in config and config['EXPORT_DIR']
    if 'CACHE_DIR' in config and config['CACHE_DIR']:
        init_cache(config['CACHE_DIR'])

    export_testbed_layouts(config['EXPORT_DIR'])




