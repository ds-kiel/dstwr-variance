#!/bin/sh
# Regenerates all figures used in the paper into $EXPORT_DIR (see .env).
# Run from the repository root. Cropped PDFs (*_cropped.pdf) require `pdfcrop`.
set -e

# Raw DS-TWR / TDoA measurements over time  -> raw_comparison.pdf
python3 scripts/export_raw_values_over_time.py

# Simulation: variance vs. response-delay ratio and NLOS bias scenarios
#   -> tdoa_sim_rmse_reponse_delay_ratio_*.pdf, simulation_bias_nlos_scenarios_*.pdf
python3 scripts/export_tdoa_simulation_response_std.py

# Simulation: NLOS bias-magnitude / probability sweep (model vs. Monte Carlo)
#   -> simulation_nlos_sweep_{twr,tdoa}.pdf, nlos_sweep_verification.csv
python3 scripts/export_nlos_sweep.py

# Testbed layout                      -> layout_trento_a.pdf
python3 scripts/export_layout.py

# Measured reception noise per link   -> rx_noise_sd_ps_trento_a.pdf
python3 scripts/export_measured_rx_noise.py

# Measured error surface over response delays -> ping_pong_3d_ds_err_*.pdf
python3 scripts/export_ping_pong_3d.py

# Model fit of the measured standard deviation -> std_fit_ping_pong_*.pdf
python3 scripts/export_ping_pong_std.py

# Double-sided vs. CFO-based single-sided comparison -> cfo_cfo_comparison_*.pdf
python3 scripts/export_ping_pong_cfo_comparison.py
