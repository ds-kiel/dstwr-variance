#!/bin/sh


# Figure 1)
python3 scripts/export_raw_values_over_time.py

# Figure 3
python3 scripts/export_tdoa_simulation_response_std.py

# Figure 4a)
python3 scripts/export_layout.py

# Figure 4b)
python3 scripts/export_measured_rx_noise.py

# Figure 4c)
python3 scripts/export_ping_pong_3d.py

# Figure 5)
python3 scripts/export_ping_pong_std.py

# Figure 6)
python3 scripts/export_ping_pong_cfo_comparison.py


