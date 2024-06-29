import os
import progressbar
import numpy as np
import json
import matplotlib.patches as mpatches

import scipy.optimize

import logs
import utility
import models
from eval_old import cached_compute_all_agg_means_and_stds, compute_all_agg_means_and_stds

from testbed import trento_a

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


def compute_passive_means_and_stds(passive_df):
    if passive_df['tdoa_est_ds'].count() != 0:
        passive_agg = passive_df.agg(
            tdoa_count=pd.NamedAgg(column='tdoa_est_ds', aggfunc="count"),
            tdoa=pd.NamedAgg(column='tdoa', aggfunc="max"),
            tdoa_est_ds_err_mean=pd.NamedAgg(column='tdoa_est_ds_err', aggfunc="mean"),
            tdoa_est_ds_mae=pd.NamedAgg(column='tdoa_est_ds_err', aggfunc=lambda x: x.abs().mean()),
            tdoa_est_ds_err_std=pd.NamedAgg(column='tdoa_est_ds_err', aggfunc="std"),
            tdoa_est_ss_init_err_mean=pd.NamedAgg(column='tdoa_est_ss_init_err', aggfunc="mean"),
            tdoa_est_ss_init_mae=pd.NamedAgg(column='tdoa_est_ss_init_err', aggfunc=lambda x: x.abs().mean()),
            tdoa_est_ss_init_err_std=pd.NamedAgg(column='tdoa_est_ss_init_err', aggfunc="std"),
            tdoa_est_ss_both_err_mean=pd.NamedAgg(column='tdoa_est_ss_both_err', aggfunc="mean"),
            tdoa_est_ss_both_mae=pd.NamedAgg(column='tdoa_est_ss_both_err', aggfunc=lambda x: x.abs().mean()),
            tdoa_est_ss_both_err_std=pd.NamedAgg(column='tdoa_est_ss_both_err', aggfunc="std"),
            tdoa_est_ss_final_err_mean=pd.NamedAgg(column='tdoa_est_ss_final_err', aggfunc="mean"),
            tdoa_est_ss_final_mae=pd.NamedAgg(column='tdoa_est_ss_final_err', aggfunc=lambda x: x.abs().mean()),
            tdoa_est_ss_final_err_std=pd.NamedAgg(column='tdoa_est_ss_final_err', aggfunc="std"),
            tdoa_est_mixed_err_mean=pd.NamedAgg(column='tdoa_est_mixed_err', aggfunc="mean"),
            tdoa_est_mixed_mae=pd.NamedAgg(column='tdoa_est_mixed_err', aggfunc=lambda x: x.abs().mean()),
            tdoa_est_mixed_err_std=pd.NamedAgg(column='tdoa_est_mixed_err', aggfunc="std")
        )
        # passive_agg.to_csv('ds-vs-cfg-pair-{}-passive-{}-passive.csv'.format(filter_pair, filter_passive_listener))
        passive_agg = passive_agg.transpose().agg('max')
        passive_dict = passive_agg.to_dict()
        return passive_dict
    return None

def compute_active_means_and_stds(active_df, skip_to_round=None):

    active_df['estimated_m'] = active_df['twr_tof_ds']
    if skip_to_round:
        active_df = active_df[active_df['round'] >= skip_to_round]


    meas_df = active_df[['pair', 'estimated_m', 'dist', 'initiator', 'responder']]

    agg = meas_df.groupby('pair').aggregate(
        count=pd.NamedAgg(column='estimated_m', aggfunc="count"),
        dist=pd.NamedAgg(column='dist', aggfunc="min"),
        estimated_m=pd.NamedAgg(column='estimated_m', aggfunc="mean"),
        initiator=pd.NamedAgg(column='initiator', aggfunc="min"),
        responder=pd.NamedAgg(column='responder', aggfunc="min"),
    )

    agg['bias'] = agg['estimated_m'] - agg['dist']

    return agg




def compute_passive_means_and_stds(passive_df, skip_to_round=None):

    passive_df['estimated_m'] = passive_df['tdoa_est_ds']

    if skip_to_round:
        passive_df = passive_df[passive_df['round'] >= skip_to_round]

    meas_df = passive_df[['pair', 'estimated_m', 'tdoa', 'initiator', 'responder']]
    agg = meas_df.groupby('pair').aggregate(
        count=pd.NamedAgg(column='estimated_m', aggfunc="count"),
        tdoa=pd.NamedAgg(column='tdoa', aggfunc="min"),
        estimated_m=pd.NamedAgg(column='estimated_m', aggfunc="mean"),
        initiator=pd.NamedAgg(column='initiator', aggfunc="min"),
        responder=pd.NamedAgg(column='responder', aggfunc="min"),
    )

    agg['bias'] = agg['estimated_m'] - agg['tdoa']

    return agg


def get_df(log, tdoa_src_dev_number):
    def proc():
        print("Processing", log, tdoa_src_dev_number)
        it = logs.gen_tdma_twr_records(trento_a, log, tdoa_src_dev_number=tdoa_src_dev_number, bias_corrected=True)
        df = pd.DataFrame.from_records(it)
        return add_df_cols(df, tdoa_src_dev_number)


    return utility.cached_dt_legacy(  # todo: this was 3
        ('export_bias_prediction_1', log, tdoa_src_dev_number), proc)



def export_bias_prediction( export_dir):
    skip_to_round = 300  # 200?
    use_bias_correction = True

    twr_df = get_df('2024-06-27_twr/job_18233.tar.gz', tdoa_src_dev_number=None)

    # remove device 5 from the list of devices as it is not working atm....
    twr_df = twr_df[twr_df['initiator'] != 5]
    twr_df = twr_df[twr_df['responder'] != 5]

    tdoa_dfs = {i: get_df('2024-06-27_twr/job_18233.tar.gz', i) for i in range(len(trento_a.devs)) if i != 5}

    for k in tdoa_dfs:
        # remove device 5 from the list of devices as it is not working atm....
        tdoa_dfs[k] = tdoa_dfs[k][tdoa_dfs[k]['initiator'] != 5]
        tdoa_dfs[k] = tdoa_dfs[k][tdoa_dfs[k]['responder'] != 5]


    twr_dict = compute_active_means_and_stds(twr_df, skip_to_round).to_dict(index=True, orient='index')
    tdoa_dicts = {k: compute_passive_means_and_stds(tdoa_dfs[k], skip_to_round).to_dict(index=True, orient='index') for k in tdoa_dfs}

    delay_a = 0.0075
    delay_b = 0.0075

    def calc_expected_td_bias(a,b,p):

        count_a = twr_dict['{}-{}'.format(a, b)]['count']
        count_b = twr_dict['{}-{}'.format(b, a)]['count']

        mean_a_b_bias = (twr_dict['{}-{}'.format(a, b)]['bias']*count_a + twr_dict['{}-{}'.format(b, a)]['bias']*count_b) / (count_a + count_b)
        # note that the twr bias does actually not matter for the model as we assume it is the same for each link
        a_b_bias_sample_mean = 0.0 #mean_a_b_bias
        b_a_bias_sample_mean = 0.0 #mean_a_b_bias


        a_p_bias_sample_mean = twr_dict['{}-{}'.format(a, p)]['bias']
        b_p_bias_sample_mean = twr_dict['{}-{}'.format(b, p)]['bias']

        return models.calc_predicted_tdoa_bias_mean(delay_b, delay_a, a_b_bias_mean=a_b_bias_sample_mean, b_a_bias_mean=b_a_bias_sample_mean, a_p_bias_mean=a_p_bias_sample_mean, b_p_bias_mean=b_p_bias_sample_mean)



    # we compare the expected tdoa bias with the actual tdoa bias

    preds = []
    actuals = []
    errors = []
    devices = []
    tdoas = []


    filter_active_dev = 3

    for p in tdoa_dicts:
        for pair in tdoa_dicts[p]:

            a, b = tuple(pair.split('-'))
            a = int(a)
            b = int(b)

            if p not in [a, b] and (filter_active_dev is None or filter_active_dev in [a, b]):
                pred = calc_expected_td_bias(a, b, p)
                actual = tdoa_dicts[p][pair]['bias']
                error = pred - actual

                preds.append(pred)
                actuals.append(actual)
                errors.append(error)
                devices.append((a,b,p))
                tdoas.append(tdoa_dicts[p][pair]['tdoa'])

    for (a,b,p), pred, actual, error in zip(devices, preds, actuals, errors):
        s = "${ini} \\rightarrow {resp}$ & ${pred_bias_mean:.3f}cm$ & ${sample_bias_mean:.3f}cm$ & ${error:.3f}cm$ \\\\"
        print(s.format(ini=a+1, resp=b+1, pred_bias_mean=round(pred, 3), sample_bias_mean=round(actual, 3), error=round(error, 3)))


    print("Count", len(preds))
    print("RMSE", np.sqrt(np.mean(np.array(errors)**2)))
    print("MAE ERROR", np.mean(np.abs(errors)))
    print("MAE BIAS", np.mean(np.abs(actuals)))
    print("Max Abs Error", np.max(np.abs(errors)))
    print("Min Abs Error", np.min(np.abs(errors)))
    print("Max Actual", np.max(np.abs(actuals)))


    # 6 0-1
    # 6 1-0
    # 6 1-3
    # 6 2-3


    # All of 6 3-X with 6-3-0 the max deviation
    # 6 0-3 expected -0.4600881173989837 actual -0.47375617984954577 error 0.013668062450562068
    # 6 1-3 expected -0.4097984239739132 actual -0.4178330944723574 error 0.008034670498444196
    # 6 2-3 expected -0.4497312955153072 actual -0.46135773613336983 error 0.011626440618062617
    # 6 3-0 expected 0.4600881173989837 actual 0.477408502324141 error -0.017320384925157306
    # 6 3-1 expected 0.4097984239739132 actual 0.40130272890988783 error 0.008495695064025366
    # 6 3-2 expected 0.4497312955153072 actual 0.44607418221240724 error 0.0036571133028999725
    # 6 3-4 expected 0.3669408518792121 actual 0.3875744904641891 error -0.02063363858497702
    # 6 4-3 expected -0.3669408518792121 actual -0.382703577579778 error 0.015762725700565916
    # Count 8
    # RMSE 0.01344574092263919
    # MAE 0.012399841393086808
    # Max Abs Error 0.02063363858497702
    # Min Abs Error 0.0036571133028999725
    # Max Actual 0.477408502324141


    # Count 120 for all devices
    # RMSE 0.02225996533904153
    # MAE 0.018540835145289578
    # Max Abs Error 0.043370486138109454
    # Min Abs Error 0.00022182169153284192
    # Max Actual 0.477408502324141

    # Count 20 for only passive device 6
    # RMSE 0.02485111307479797
    # MAE 0.02093110328928046
    # Max Abs Error 0.043370486138109454
    # Min Abs Error 0.0021931930191732008
    # Max Actual 0.477408502324141



    #    gb = df.groupby('pair').agg(
    #         count=pd.NamedAgg(column='tdoa_est_ds', aggfunc="count"),
    #         dist=pd.NamedAgg(column='dist', aggfunc="min"),
    #         tdoa=pd.NamedAgg(column='tdoa', aggfunc="min"),
    #         phase_dist_err_mean=pd.NamedAgg(column='phase_dist_err', aggfunc="mean"),
    #         phase_dist_err_std=pd.NamedAgg(column='phase_dist_err', aggfunc="std"),
    #         twr_tof_ds_err_mean=pd.NamedAgg(column='twr_tof_ds_err', aggfunc="mean"),
    #         twr_tof_ds_err_std=pd.NamedAgg(column='twr_tof_ds_err', aggfunc="std"),
    #         twr_tof_ss_err_mean=pd.NamedAgg(column='twr_tof_ss_err', aggfunc="mean"),
    #         twr_tof_ss_err_std=pd.NamedAgg(column='twr_tof_ss_err', aggfunc="std"),
    #         twr_tof_ss_reverse_err_mean=pd.NamedAgg(column='twr_tof_ss_reverse_err', aggfunc="mean"),
    #         twr_tof_ss_reverse_err_std=pd.NamedAgg(column='twr_tof_ss_reverse_err', aggfunc="std"),
    #         tdoa_est_ds_err_mean=pd.NamedAgg(column='tdoa_est_ds_err', aggfunc="mean"),
    #         tdoa_est_ds_err_std=pd.NamedAgg(column='tdoa_est_ds_err', aggfunc="std"),
    #         tdoa_est_ss_init_err_mean=pd.NamedAgg(column='tdoa_est_ss_init_err', aggfunc="mean"),
    #         tdoa_est_ss_init_err_std=pd.NamedAgg(column='tdoa_est_ss_init_err', aggfunc="std"),
    #         tdoa_est_ss_both_err_mean=pd.NamedAgg(column='tdoa_est_ss_both_err', aggfunc="mean"),
    #         tdoa_est_ss_both_err_std=pd.NamedAgg(column='tdoa_est_ss_both_err', aggfunc="std"),
    #         tdoa_est_ss_final_err_mean=pd.NamedAgg(column='tdoa_est_ss_final_err', aggfunc="mean"),
    #         tdoa_est_ss_final_err_std=pd.NamedAgg(column='tdoa_est_ss_final_err', aggfunc="std"),
    #         tdoa_est_mixed_err_mean=pd.NamedAgg(column='tdoa_est_mixed_err', aggfunc="mean"),
    #         tdoa_est_mixed_err_std=pd.NamedAgg(column='tdoa_est_mixed_err', aggfunc="std")
    #     )
    #
    #
    #
    #     gb.plot.bar(y=['twr_tof_ds_err_std', 'phase_dist_err_std'])
    #
    #     plt.show()


if __name__ == '__main__':
    config = load_env_config()
    load_plot_defaults()
    assert 'EXPORT_DIR' in config and config['EXPORT_DIR']
    if 'CACHE_DIR' in config and config['CACHE_DIR']:
        init_cache(config['CACHE_DIR'])

    export_bias_prediction(config['EXPORT_DIR'])




