
import numpy as np
from base import SPEED_OF_LIGHT_IN_AIR


# class DSTWRParams:
#     def __init__(self, delay_a=0.0, delay_b=0.0, a_b_std=0.0, b_a_std=0.0, a_b_bias=0.0, b_a_bias=0.0):
#         self.delay_a = delay_a
#         self.delay_b = delay_b
#         self.a_b_std = a_b_std
#         self.b_a_std = b_a_std
#         self.a_b_bias = a_b_bias
#         self.b_a_bias = b_a_bias
#         self.delay_ratio = self.delay_a / (self.delay_b + self.delay_a) # we ignore the time of flight here ;)
#
#     def expected_tof_bias(self):
#         pass
#
#     def expected_tof_std(self):
#         pass
#
# class DSTDoAParams:
#     def __init__(self, ds_twr_params, a_p_std=0.0, b_p_std=0.0, a_p_bias=0.0, b_p_bias=0.0):
#         self.ds_twr_params = ds_twr_params
#         self.a_p_std = a_p_std
#         self.b_p_std = b_p_std
#         self.a_p_bias = a_p_bias
#         self.b_p_bias = b_p_bias
#
#     def expected_tdoa_bias(self):
#         pass
#
#     def expected_tdoa_std(self):
#         pass


def calc_predicted_tof_bias_mean(delay_b, delay_a, a_b_bias_mean=0.0, b_a_bias_mean=0.0):
    return 0.5*b_a_bias_mean + 0.5 * a_b_bias_mean # quite an easy calculation actually

def calc_predicted_tdoa_bias_mean(delay_b, delay_a, a_b_bias_mean=0.0, b_a_bias_mean=0.0, a_p_bias_mean=0.0, b_p_bias_mean=0.0):
    return 0.5*b_a_bias_mean - 0.5 * a_b_bias_mean + a_p_bias_mean - b_p_bias_mean


def calc_predicted_tof_std(delay_b, delay_a, a_b_std, b_a_std, a_b_bias=0.0, b_a_bias=0.0):
    return np.sqrt(
        (0.5 * b_a_std) ** 2
        + (0.5 * (delay_b / (delay_a + delay_b)) * a_b_std) ** 2
        + (0.5 * (1 - (delay_b / (delay_a + delay_b))) * a_b_std) ** 2
    )

def calc_predicted_tdoa_std(delay_b, delay_a, a_b_std, b_a_std, a_p_std, b_p_std):

    comb_delay = delay_a + delay_b

    return np.sqrt(
        (0.5 * b_a_std) ** 2
        + (0.5 * a_b_std * (delay_b /comb_delay -1)) ** 2
        + (0.5 * a_b_std * (delay_b /comb_delay)) ** 2
        + (a_p_std * ( 1 - delay_b /comb_delay)) ** 2
        + b_p_std ** 2
        + (a_p_std * (delay_b /comb_delay)) ** 2
    )


def calc_predicted_tof_std_navratil(a_b_std, b_a_std, delay_b, delay_a, distance=10.0): # TODO: this value is from the simulation, i.e., the true range

    if a_b_std != b_a_std:
        return None # we cannot predict in this model

    sigma = a_b_std

    r = distance
    tof = r / SPEED_OF_LIGHT_IN_AIR

    t_b1 = delay_b
    t_a2 = delay_a
    t_a1 = t_b1 + 2*tof
    t_b2 = t_a2 + 2*tof

    sigma_mu = (delay_b+delay_a+3*tof)*2

    var = sigma*sigma*(
            (
                    2.0*(
                        t_b2 * t_b2
                        + t_a1*t_a1
                        + t_b1*t_b1
                        + t_a2*t_a2
                        + t_a1*t_a2
                        + t_b1*t_b2
                    )
                    #+ (4.0 * r * r / (SPEED_OF_LIGHT_IN_AIR * SPEED_OF_LIGHT_IN_AIR))
            ) / (sigma_mu*sigma_mu)
           )

    return np.sqrt(var)


# the noise map is a simple map from (tx, rx) to a tuple of mean and std and an optional function to sample from the underlying distribution, Gaussian by default
def rx_noise_map_entry(rx_noise_map, tx, rx):
    if isinstance(rx_noise_map, dict):
        if "{}-{}".format(tx, rx) in rx_noise_map:
            return rx_noise_map["{}-{}".format(tx, rx)]
        else:
            return rx_noise_map[(tx, rx)]
    else:
        return rx_noise_map

def rx_noise_map_mean(rx_noise_map, tx, rx):
    return rx_noise_map_entry(rx_noise_map, tx, rx)[0]

def rx_noise_map_std(rx_noise_map, tx, rx):
    return rx_noise_map_entry(rx_noise_map, tx, rx)[1]

def rx_noise_map_sample(rx_noise_map, tx, rx):
    entry = rx_noise_map_entry(rx_noise_map, tx, rx)

    if len(entry) == 2:
        return np.random.normal(loc=entry[0], scale=entry[1])
    elif len(entry) == 3 and callable(entry[2]):
        return entry[2](**{'tx': tx, 'rx': rx, 'loc': entry[0], 'scale': entry[1]})
