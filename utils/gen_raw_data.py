import argparse
import logging
from itertools import cycle
from pathlib import Path
from pprint import pformat
from typing import List

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from pandas._libs.tslibs.timestamps import Timestamp
from ruptures.utils import draw_bkps
from timeseries_generator import (Generator, HolidayFactor, LinearTrend,
                                  RandomFeatureFactor, SinusoidalFactor,
                                  WeekdayFactor, WhiteNoise)
from timeseries_generator.external_factors import (CountryGdpFactor,
                                                   EUIndustryProductFactor)

current_dir = Path(__file__).parent

logger = logging.getLogger(__name__)
logger_console = logging.StreamHandler()
logger_formatter = logging.Formatter('%(levelname)-8s [%(filename)s] %(message)s')
logger_console.setFormatter(logger_formatter)
logger.addHandler(logger_console)
logger.setLevel(logging.INFO)


def pw_rand_f1_f2_wavy(n_samples=200, n_bkps=3, noise_std=None, seed=120):
    """Return a 1D piecewise wavy signal and the associated changepoints.

    Args:
        n_samples (int, optional): signal length
        n_bkps (int, optional): number of changepoints
        noise_std (float, optional): noise std. If None, no noise is added
        seed (int): random seed, the frequence 1 and frequence 2 also based on seed

    Returns:
        tuple: signal of shape (n_samples), list of breakpoints
    """
    # breakpoints
    bkps = draw_bkps(n_samples, n_bkps, seed=seed)
    # we create the signal
    rng = np.random.default_rng(seed=seed)
    f1, f2 = rng.uniform(low=0, high=1, size=(2,2))
    freqs = np.zeros((n_samples, 2))
    for sub, val in zip(np.split(freqs, bkps[:-1]), cycle([f1, f2])):
        sub += val
    tt = np.arange(n_samples)

    # DeprecationWarning: Calling np.sum(generator) is deprecated
    # Use np.sum(np.from_iter(generator)) or the python sum builtin instead.
    signal = np.sum([np.sin(2 * np.pi * tt * f) for f in freqs.T], axis=0)

    if noise_std is not None:
        noise = rng.normal(scale=noise_std, size=signal.shape)
        signal += noise

    return signal, bkps


def pw_adapt_cov_linear_combine(n_samples=200, n_features=1, n_bkps=3, noise_std=0, cov_loc=50, seed=None):
    """Return piecewise linear signal and the associated changepoints.

    Args:
        n_samples (int, optional): signal length
        n_features (int, optional): number of covariates
        n_bkps (int, optional): number of change points
        noise_std (float, optional): noise std. If None, no noise is added
        seed (int): random seed
    Returns:
        tuple: signal of shape (n_samples, n_features+1), list of breakpoints
    """
    rng = np.random.default_rng(seed=seed)
    covar = rng.normal(loc=cov_loc, scale=abs(cov_loc*(noise_std/100)), size=(n_samples, n_features))
    linear_coeff, bkps = rpt.pw_constant(
            n_samples=n_samples,
            n_bkps=n_bkps,
            n_features=n_features,
            noise_std=None,
            seed=seed,
            )
    var = np.sum(linear_coeff * covar, axis=1)
    if noise_std != 0:
        var += rng.normal(scale=abs(var.mean()*(noise_std/100)), size=var.shape)
    signal = np.c_[var, covar]

    return signal, bkps


def nike_ts(n_samples: int = 200, country: str = "United Kingdom", start_date: str = "2010/1/1"):
    """Return a time-series with some calender factors(week, holiday...) and external factors.

    Args:
        n_samples (int, optional): signal length
        country (str, optional): country for setting factor
        start_date (str, optional): string of date

    Returns:
        tuple: signal of shape (n_samples)
    """
    start_date = Timestamp(start_date)
    c_gdp_factor = CountryGdpFactor(country_list=[country])
    eu_industry_product_factor = EUIndustryProductFactor()
    holiday_factor = HolidayFactor(
        holiday_factor=1.5,
        special_holiday_factors={
            "Christmas Day": 3.
        },
        country_list=[country.replace(" ", "_")]
    )
    weekday_factor = WeekdayFactor(
        col_name="weekend_boost_factor",
        factor_values={0: 1.3, 4: 1.15}  # Here we assign a factor of 1.15 to Friday, and 1.3 to Sat/Sun
    )
    product_seasonal_components = SinusoidalFactor(
        feature="basis_trend_seasonality",
        col_name="basis_trend_seasonal_factor",
        feature_values={
            "year": {
                "wavelength": 365.,
                "amplitude": 0.2,
                "phase": 365/4,
                "mean": 1.
            },
        }
    )
    features_dict = {
            "country": [country.lower().replace(" ", "").replace("_", "")],
            "basis_trend_seasonality": ["year"]
        }
    g: Generator = Generator(
        factors={
            c_gdp_factor,
            eu_industry_product_factor,
            holiday_factor,
            weekday_factor,
            product_seasonal_components,
        },
        features=features_dict,
        date_range=pd.date_range(start=start_date, periods=n_samples),
        base_value=10000
    )
    df = g.generate()

    return df['value'].values


def specific_ts(df_path: Path, ts_name: str):
    df = pd.read_csv(df_path)
    return df[ts_name].values


def gen_pw_constant_data(args):
    """
    Generate piecewise constant data.
    ref:
        - https://centre-borelli.github.io/ruptures-docs/code-reference/datasets/pw_constant-reference/
        - https://centre-borelli.github.io/ruptures-docs/user-guide/datasets/pw_constant/
    """
    n, dim = args.time_len, args.dim  # time_length(number of samples), number of variables(dimension)
    n_bkps, sigma = args.n_bkps, args.noise_std  # number of change points, noise standart deviation
    signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma, delta=(1, 500), seed=0)
    signal = signal+abs(signal.min())+1
    dates = pd.to_datetime(range(n), unit='D', origin=pd.Timestamp('now'))  # create a DatetimeIndex with interval of one day
    var_names = [f'var_{i}' for i in range(dim)]
    df = pd.DataFrame(signal, index=dates, columns=var_names)
    df.index.name = "Date"
    logger.info(f"Generated piecewise constant data with shape {df.shape} and {n_bkps} change points.")
    logger.info(f"piecewiase constant data[:5, :5]:\n{df.iloc[:5, :5]}")
    if args.save_data:
        save_dir = set_save_dir(args)
        df.to_csv(save_dir/f'pw_constant-bkps{n_bkps}-noise_std{sigma}.csv')

    return signal, bkps


def gen_pw_linear_combine_data(args):
    """
    Generate piecewise linear data.
    ref:
        - https://centre-borelli.github.io/ruptures-docs/code-reference/datasets/pw_linear-reference/#ruptures.datasets.pw_linear.pw_linear
        - https://centre-borelli.github.io/ruptures-docs/user-guide/datasets/pw_linear/
    """
    n, dim = args.time_len, args.dim  # time_length(number of samples), number of variables(dimension)
    n_bkps, sigma = args.n_bkps, args.noise_std  # number of change points, noise standart deviation
    signal, bkps = pw_adapt_cov_linear_combine(n, dim, n_bkps, noise_std=sigma, seed=0)
    dates = pd.to_datetime(range(n), unit='D', origin=pd.Timestamp('now'))  # create a DatetimeIndex with interval of one day
    var_names = ['var_linear_sum']+[f'var_{i}' for i in range(dim)]
    df = pd.DataFrame(signal, index=dates, columns=var_names)
    df.index.name = "Date"
    logger.info(f"Generated piecewise linear_combine data with shape {df.shape} and {n_bkps} change points.")
    logger.info(f"piecewiase linear_combine data[:5, :5]:\n{df.iloc[:5, :5]}")
    if args.save_data:
        save_dir = set_save_dir(args)
        df.to_csv(save_dir/f'pw_linear_combine-bkps{n_bkps}-noise_std{sigma}.csv')

    return signal, bkps


def gen_pw_wave_const_data(args, seed=120):
    """
    Generate piecewise wave * constant data.
    """
    n, dim = args.time_len, args.dim  # time_length(number of samples), number of variables(dimension)
    n_bkps, sigma = args.n_bkps, args.noise_std  # number of change points, noise standart deviation
    wave_signal, bkps = pw_rand_f1_f2_wavy(n, n_bkps, noise_std=0, seed=seed)
    const_signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=0, delta=(1, 500), seed=0)
    signal = wave_signal.reshape(n, 1)*const_signal
    rng = np.random.default_rng(seed=0)
    for i, const in enumerate(const_signal[0, ::]):
        noise = rng.normal(scale=abs(const*(sigma/100)), size=n)
        signal[::, i] = signal[::, i]+noise
    signal = signal+abs(signal.min())+1
    dates = pd.to_datetime(range(n), unit='D', origin=pd.Timestamp('now'))  # create a DatetimeIndex with interval of one day
    var_names = [f'var_{i}' for i in range(dim)]
    df = pd.DataFrame(signal, index=dates, columns=var_names)
    df.index.name = "Date"
    logger.info(f"Generated piecewise wave_constant data with shape {df.shape} and {n_bkps} change points.")
    logger.info(f"Piecewise wave_constant data[:5, :5]:\n{df.iloc[:5, :5]}")
    if args.save_data:
        save_dir = set_save_dir(args)
        df.to_csv(save_dir/f'pw_wave_const-bkps{n_bkps}-noise_std{sigma}.csv')

    return signal, bkps


def gen_pw_wave_t_shift_data(args, seed=120):
    """
    Generate piecewise wave with time shift data.
    """
    n, dim = args.time_len, args.dim  # time_length(number of samples), number of variables(dimension)
    n_bkps, sigma = args.n_bkps, args.noise_std  # number of change points, noise standart deviation
    t_shift_intv = int(n/50)
    rng = np.random.default_rng(seed=0)
    long_wave_signal, bkps = pw_rand_f1_f2_wavy(n+(dim*2*t_shift_intv), n_bkps, noise_std=0, seed=0)
    one_signal = np.ones((n, dim))
    signal = np.zeros(one_signal.shape)
    start_idx_list = rng.integers(low=0, high=dim*2, size=dim)*t_shift_intv
    for i, start_idx in enumerate(start_idx_list):
        wave_signal = long_wave_signal[start_idx:start_idx+n]
        signal[::, i] = wave_signal*one_signal[::, i]
        noise = rng.normal(scale=abs(1*(sigma/100)), size=n)
        signal[::, i] = signal[::, i]+noise
    signal = signal+abs(signal.min())+1
    dates = pd.to_datetime(range(n), unit='D', origin=pd.Timestamp('now'))  # create a DatetimeIndex with interval of one day
    var_names = [f'var_{i}' for i in range(dim)]
    df = pd.DataFrame(signal, index=dates, columns=var_names)
    df.index.name = "Date"
    logger.info(f"Generated piecewise wave_t_shift data with shape {df.shape} and {n_bkps} change points.")
    logger.info(f"Piecewise wave_t_shift data[:5, :5]:\n{df.iloc[:5, :5]}")
    if args.save_data:
        save_dir = set_save_dir(args)
        df.to_csv(save_dir/f'pw_wave_t_shift-bkps{n_bkps}-noise_std{sigma}.csv')

    return signal, bkps


def gen_pw_wave_multiply_linear_reg_data(args, seed=120):
    """
    Generate piecewise wave * linear_regression data.
    """
    n, dim = args.time_len, args.dim  # time_length(number of samples), number of variables(dimension)
    n_bkps, sigma = args.n_bkps, args.noise_std  # number of change points, noise standart deviation
    rng = np.random.default_rng(seed=0)
    wave_signal, bkps = pw_rand_f1_f2_wavy(n, n_bkps, noise_std=0, seed=seed)
    const_signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=0, delta=(1, 500), seed=0)
    linear_reg_signal = np.zeros(const_signal.shape)
    tt = np.arange(n)
    for i, (sub_linear_reg_signal, reg_coef) in enumerate(zip(np.split(linear_reg_signal, dim, axis=1), rng.uniform(low=-2, high=2, size=dim))):
        sub_linear_reg_signal += (tt*reg_coef+const_signal[::, i]).reshape(-1, 1)
    signal = wave_signal.reshape(n, 1)*linear_reg_signal
    for i, const_linear_reg_mean in enumerate(linear_reg_signal.mean(axis=0)):
        noise = rng.normal(scale=abs(const_linear_reg_mean*(sigma/100)), size=n)
        signal[::, i] = signal[::, i]+noise
    signal = signal+abs(signal.min())+1
    dates = pd.to_datetime(range(n), unit='D', origin=pd.Timestamp('now'))  # create a DatetimeIndex with interval of one day
    var_names = [f'var_{i}' for i in range(dim)]
    df = pd.DataFrame(signal, index=dates, columns=var_names)
    df.index.name = "Date"
    logger.info(f"Generated piecewise wave_multiply_linear_regression data with shape {df.shape} and {n_bkps} change points.")
    logger.info(f"Piecewise wave_multiply_linear_reg data[:5, :5]:\n{df.iloc[:5, :5]}")
    if args.save_data:
        save_dir = set_save_dir(args)
        df.to_csv(save_dir/f'pw_wave_multiply_linear_reg-bkps{n_bkps}-noise_std{sigma}.csv')

    return signal, bkps


def gen_pw_wave_add_linear_reg_data(args, seed=120):
    """
    Generate piecewise wave + linear_regression data.
    """
    n, dim = args.time_len, args.dim  # time_length(number of samples), number of variables(dimension)
    n_bkps, sigma = args.n_bkps, args.noise_std  # number of change points, noise standart deviation
    rng = np.random.default_rng(seed=0)
    wave_signal, bkps = pw_rand_f1_f2_wavy(n, n_bkps, noise_std=0, seed=seed)
    const_signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=0, delta=(1, 500), seed=0)
    wave_const_signal = wave_signal.reshape(n, 1)*const_signal
    linear_reg_signal = np.zeros(const_signal.shape)
    tt = np.arange(n)
    for i, (sub_linear_reg_signal, reg_coef) in enumerate(zip(np.split(linear_reg_signal, dim, axis=1), rng.uniform(low=-2, high=2, size=dim))):
        sub_linear_reg_signal += (tt*reg_coef+const_signal[::, i]).reshape(-1, 1)
    signal = wave_const_signal+linear_reg_signal
    for i, const_linear_reg_mean in enumerate(linear_reg_signal.mean(axis=0)):
        noise = rng.normal(scale=abs(const_linear_reg_mean*(sigma/100)), size=n)
        signal[::, i] = signal[::, i]+noise
    signal = signal+abs(signal.min())+1
    dates = pd.to_datetime(range(n), unit='D', origin=pd.Timestamp('now'))  # create a DatetimeIndex with interval of one day
    var_names = [f'var_{i}' for i in range(dim)]
    df = pd.DataFrame(signal, index=dates, columns=var_names)
    df.index.name = "Date"
    logger.info(f"Generated piecewise wave_add_linear_regression data with shape {df.shape} and {n_bkps} change points.")
    logger.info(f"Piecewise wave_add_linear_reg data[:5, :5]:\n{df.iloc[:5, :5]}")
    if args.save_data:
        save_dir = set_save_dir(args)
        df.to_csv(save_dir/f'pw_wave_add_linear_reg-bkps{n_bkps}-noise_std{sigma}.csv')

    return signal, bkps


def gen_pw_wave_linear_combine_data(args):
    """
    Generate piecewise wave * linear data.
    """
    n, dim = args.time_len, args.dim  # time_length(number of samples), number of variables(dimension)
    n_bkps, sigma = args.n_bkps, args.noise_std  # number of change points, noise standart deviation
    wave_signal, bkps = pw_rand_f1_f2_wavy(n, n_bkps, noise_std=0)
    linear_signal, bkps = pw_adapt_cov_linear_combine(n, dim, n_bkps, noise_std=sigma, seed=0)
    signal = wave_signal.reshape(n, 1)*linear_signal
    dates = pd.to_datetime(range(n), unit='D', origin=pd.Timestamp('now'))  # create a DatetimeIndex with interval of one day
    var_names = ['var_linear_sum']+[f'var_{i}' for i in range(dim)]
    df = pd.DataFrame(signal, index=dates, columns=var_names)
    df.index.name = "Date"
    logger.info(f"Generated piecewise wave_linear_combine data with shape {df.shape} and {n_bkps} change points.")
    logger.info(f"piecewiase wave_linear_combine data[:5, :5]:\n{df.iloc[:5, :5]}")
    if args.save_data:
        save_dir = set_save_dir(args)
        df.to_csv(save_dir/f'pw_wave_linear_combine-bkps{n_bkps}-noise_std{sigma}.csv')

    return signal, bkps


def gen_leader_signal(args):
    n, dim, n_bkps = args.time_len, args.dim, args.n_bkps  # time_length(number of samples), number of variables(dimension), number of change points
    rng = np.random.default_rng(seed=0)
    seg_len = int(n/(n_bkps+1))
    n = n-(n % seg_len)  # remove remainder
    basis_m_list = rng.uniform(low=-10, high=10, size=(n_bkps+1, 1))
    basis_b = rng.uniform(low=0, high=10, size=1)
    tt = np.arange(seg_len)
    basis_trend = np.zeros(n)
    for i, basis_m in enumerate(basis_m_list):
        basis_trend[i*seg_len:(i+1)*seg_len] = basis_m*tt+basis_b
        basis_b = basis_trend[(i+1)*seg_len-1]
    basis_trend_mean = basis_trend.mean()
    if args.basis_type == "pw_rand_wavy":
        wave_signal, bkps = pw_rand_f1_f2_wavy(n_samples=n, n_bkps=0, noise_std=0, seed=seed)
        leader_signal = (wave_signal*basis_trend_mean)+basis_trend
    elif args.basis_type == "nike_ts":
        nike_signal, bkps = nike_ts(n_samples=n), [n]
        leader_signal = nike_signal+basis_trend
    elif args.basis_type == "specific_ts":
        leader_signal, bkps = specific_ts(args.specific_ts_path, args.specific_ts_var), [n]

    return leader_signal, bkps


def exec_post_processing(signal: np.array, noise_scale: float):
    rng = np.random.default_rng(seed=0)
    n, dim = signal.shape
    noise_scale = noise_scale/100
    for sub_signal in np.split(signal, dim, axis=1):
        # add noise after create signal
        standard_noise = rng.normal(loc=0, scale=0.5, size=n).reshape(-1, 1)
        scale_noise =  noise_scale*sub_signal*standard_noise
        sub_signal += scale_noise
    signal = signal+abs(signal.min())+1
    dates = pd.to_datetime(range(n), unit='D', origin=pd.Timestamp('now'))  # create a DatetimeIndex with interval of one day
    var_names = [f'var_{i}' for i in range(dim)]
    df = pd.DataFrame(signal, index=dates, columns=var_names)
    df.index.name = "Date"

    return signal, df


def gen_linear_reg_cluster_data(args, seed=120):
    """
    Generate cluster data whose instances are linear correlation to leader_signal.
    Specifing `args.n_bkps` to decide the number of change-point of  leader_signal.
    """
    dim, noise_scale, n_bkps = args.dim, args.noise_scale, args.n_bkps
    rng = np.random.default_rng(seed=0)
    leader_signal, bkps = gen_leader_signal(args)
    no_noise_signal = np.zeros((args.time_len, dim))
    for i, (sub_signal, (reg_coef, reg_bias)) in enumerate(zip(np.split(no_noise_signal, dim, axis=1), rng.uniform(low=-10, high=10, size=(dim, 2)))):
        if i == 0:
            sub_signal += leader_signal.reshape(-1, 1)
        else:
            sub_signal += (reg_coef*leader_signal+reg_bias).reshape(-1, 1)  # create follower_signal that has linear correlation to leader_signal
    signal, df = exec_post_processing(signal=no_noise_signal, noise_scale=noise_scale)
    logger.info(f"Generated linear_regression cluster data with shape {df.shape} and basis_type {args.basis_type} and {n_bkps} change points.")
    logger.info(f"Linear regression cluster data[:5, :5]:\n{df.iloc[:5, :5]}")
    if args.save_data:
        save_dir = set_save_dir(args)
        df.to_csv(save_dir/f'linear_reg_one_cluster-{args.basis_type}-bkps{n_bkps}-noise_scale{noise_scale}.csv')

    return signal, bkps


def gen_power_cluster_data(args, seed=120):
    """
    Generate cluster data whose instances are power ($$X_n = m_n*X_{leader}+b_n$$)(non-linear) correlation to leader_signal
    Specifing `args.n_bkps` to decide the number of change-point of leader_signal.
    """
    dim, noise_scale, n_bkps = args.dim, args.noise_scale, args.n_bkps
    basis_type, power = args.basis_type, args.power
    rng = np.random.default_rng(seed=0)
    leader_signal, bkps = gen_leader_signal(args)
    no_noise_signal = np.zeros((args.time_len, dim))
    for i, (sub_signal, (reg_coef, reg_bias)) in enumerate(zip(np.split(no_noise_signal, dim, axis=1), rng.uniform(low=-10, high=10, size=(dim, 2)))):
        if i == 0:
            sub_signal += leader_signal.reshape(-1, 1)
        else:
            sub_signal += (reg_coef*(leader_signal**power)+reg_bias).reshape(-1, 1)  # create sub_variable that has linear correlation to power_2 of leader_signal
    signal, df = exec_post_processing(signal=no_noise_signal, noise_scale=noise_scale)
    logger.info(f"Generated power_{power} cluster data with shape {df.shape} and basis_type {basis_type} and {n_bkps} change points.")
    logger.info(f"Power {power} cluster data[:5, :5]:\n{df.iloc[:5, :5]}")
    if args.save_data:
        save_dir = set_save_dir(args)
        df.to_csv(save_dir/f'power_{str(power).replace(".", "_")}_one_cluster-{basis_type}-bkps{n_bkps}-noise_scale{noise_scale}.csv')

    return signal, bkps


def gen_t_shift_cluster_data(args, seed=120):
    """
    Generate  cluster data whose instances has time shift  to leader_signal.
    Specifing `args.n_bkps` to decide the number of change-point of  leader_signal.
    """
    dim, noise_scale, n_bkps = args.dim, args.noise_scale, args.n_bkps
    start_idx_list = np.array([0]+[i for i in range(100) if i % 7 != 0])[:dim]
    args.time_len += start_idx_list[-1]
    leader_signal, bkps = gen_leader_signal(args)
    args.time_len -= start_idx_list[-1]
    no_noise_signal = np.zeros((args.time_len, dim))
    for i, (sub_signal, start_idx) in enumerate(zip(np.split(no_noise_signal, dim, axis=1), start_idx_list)):
        sub_signal += leader_signal[start_idx:start_idx+args.time_len].reshape(-1, 1)
    signal, df = exec_post_processing(signal=no_noise_signal, noise_scale=noise_scale)
    logger.info(f"Generated t_shift_one_cluster data with shape {df.shape} and basis_type {args.basis_type} and {n_bkps} change points.")
    logger.info(f"Time shift cluster data[:5, :5]:\n{df.iloc[:5, :5]}")
    if args.save_data:
        save_dir = set_save_dir(args)
        df.to_csv(save_dir/f't_shift_one_cluster-{args.basis_type}-bkps{n_bkps}-noise_scale{noise_scale}.csv')

    return signal, bkps


def gen_multi_cluster_data(args, gen_data_func):
    """
    Generate multiple cluseter data.
    Construct the clusters by data that produce by `gen_data_func`.
    """
    gen_data_func_name = gen_data_func.__name__[7:-5]
    n, dim = args.time_len, args.dim  # time_length(number of samples), number of variables(dimension)
    n_bkps, sigma = args.n_bkps, args.noise_std  # number of change points, noise standart deviation
    n_clusters = args.n_clusters  # number of clusters
    signal = np.zeros((n, n_clusters*dim))
    for sub, cluster_idx in zip(np.split(signal, n_clusters, axis=1), range(n_clusters)):
        cluster_signal, bkps = gen_data_func(args, seed=cluster_idx)
        sub += cluster_signal
    dates = pd.to_datetime(range(n), unit='D', origin=pd.Timestamp('now'))  # create a DatetimeIndex with interval of one day
    var_names = [f'cluster_{cluster_idx}_var_{i}' for cluster_idx in range(n_clusters) for i in range(dim)]
    df = pd.DataFrame(signal, index=dates, columns=var_names)
    df.index.name = "Date"
    logger.info(f"Generated clusters_{n_clusters} piecewise {gen_data_func_name} data with shape {df.shape} and {n_bkps} change points.")
    logger.info(f"clusters_{n_clusters} piecewiase {gen_data_func_name} data[:5, :5]:\n{df.iloc[:5, :5]}")
    if args.save_data:
        save_dir = set_save_dir(args)
        df.to_csv(save_dir/f'cluster_{n_clusters}-pw_{gen_data_func_name}-bkps{n_bkps}-noise_std{sigma}.csv')


def merge_multi_cluster_data(df_path_list: List[Path]):
    """
    Generate multiple cluseter data.
    Construct the clusters by specific dataframes.
    """
    n_clusters = len(df_path_list)
    df = pd.DataFrame()
    for i, df_path in enumerate(df_path_list):
        tmp_df = pd.read_csv(df_path, index_col=["Date"])
        tmp_df.columns = [f"cluster_{i}_{col}" for col in tmp_df.columns]
        df = pd.concat((df, tmp_df), axis=1)
    logger.info(f"Merge specific dataframes to form a clusters_{n_clusters} data with shape {df.shape}")
    logger.info(f"specific dataframes:{[path.stem for path in df_path_list]}")
    logger.info(f"Merge multi cluster data[:5, :5]:\n{df.iloc[:5, :5]}")
    if args.save_data:
        save_dir = set_save_dir(args)
        df.to_csv(save_dir/f'merge_cluster_{n_clusters}.csv')

    return df.values


def set_save_dir(args):
    """
    Set the save directory.
    """
    if args.n_clusters:
        save_dir = current_dir/f'../dataset/synthetic/dim{args.n_cluster*args.dim}'
    elif args.merge_df_paths:
        dim = sum(len(pd.read_csv(Path(path), index_col=["Date"]).columns) for path in args.merge_df_paths)
        dir_tree = "/".join([str(Path(path).stem) for path in args.merge_df_paths])
        save_dir = current_dir/f'../dataset/synthetic/dim{dim}/merge_cluster_ts/{dir_tree}'
    elif args.specific_ts_path:
        save_dir = current_dir/f'../dataset/synthetic/dim{args.dim}/basis_specific_ts/{Path(args.specific_ts_path).stem}'
    else:
        save_dir = current_dir/f'../dataset/synthetic/dim{args.dim}'
    save_dir.mkdir(parents=True, exist_ok=True)

    return save_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate raw data.')
    parser.add_argument('--data_type', type=str, default=['pw_constant'], nargs='+',
                        choices=['pw_constant', 'pw_linear_combine', 'pw_wave_const', 'pw_wave_linear_combine',
                                 'pw_wave_multiply_linear_reg', 'pw_wave_add_linear_reg', 'pw_wave_t_shift',
                                 'linear_reg_cluster', 'power_cluster', 't_shift_cluster',
                                 'multi_cluster', "merge_multi_cluster"],
                        help='Type of data to generate. (default: pw_constant)')
    parser.add_argument('--time_len', type=int, default=2600, help='Input time length. (default: 2600)')
    parser.add_argument('--dim', type=int, default=30, help='Input dimension(number of variable). (default: 30)')
    parser.add_argument('--noise_scale', type=int, default=3, help='Input noise scale in the form of percentage integer. (default: 3))')
    parser.add_argument('--n_bkps', type=int, default=0, help='Input number of change points. (default: 0)')
    parser.add_argument('--n_clusters', type=int, default=0, help='Input number of clusters. (default: 0)')
    parser.add_argument('--basis_type', type=str, default='nike_ts', nargs='?',
                        choices=['nike_ts', 'pw_rand_wavy', 'specific_ts'],
                        help='Type of basis_signal to generate. (default: nike_ts)')
    parser.add_argument('--power', type=float, default=None, help='Input power of basis_signal.')
    parser.add_argument('--specific_ts_path', type=str, default=None,
                        help='Input path of specific_ts. (default: None)')
    parser.add_argument('--specific_ts_var', type=str, default=None,
                        help='Input variable name of specific_ts. (default: None)')
    parser.add_argument('--merge_df_paths', type=str, default=None, nargs='+',
                        help='Input paths of dataframes to merge. (default: None)')
    parser.add_argument("--save_data", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="input --save_data to save raw data")
    args = parser.parse_args()
    assert bool("power_cluster" in args.data_type) == bool(args.power), "`power` should be set when `data_type` is 'power_cluster'"
    assert bool("multi_cluster" in args.data_type) == bool(args.n_clusters >= 2), "`n_clusters` should be set when `data_type` is 'multi_cluster'"
    assert bool("multi_cluster" in args.data_type) == bool(len(args.data_type) == 2), "`data_type` should contain another generate data setting when 'multi_cluster' is set"
    assert bool("specific_ts" in args.basis_type) == bool(args.specific_ts_path), "`specific_ts_path` should be set when `basis_type` is 'specific_ts'"
    assert bool("specific_ts" in args.basis_type) == bool(args.specific_ts_var), "`specific_ts_var` should be set when `basis_type` is 'specific_ts'"
    assert bool("merge_multi_cluster" in args.data_type) == bool(args.merge_df_paths), "`merge_df_paths` should be set when `data_type` is 'merge_multi_cluster'"
    logger.info(pformat(vars(args), indent=1, width=100, compact=True))

    if 'multi_cluster' in args.data_type:
        args.data_type.remove('multi_cluster')
        func = locals()[f'gen_{args.data_type[0]}_data']
        gen_multi_cluster_data(args, func)
    elif 'merge_multi_cluster' in args.data_type:
        df_path_list = [Path(path) for path in args.merge_df_paths]
        merge_multi_cluster_data(df_path_list)
    else:
        if 'pw_constant' in args.data_type:
            gen_pw_constant_data(args)
        if 'pw_linear_combine' in args.data_type:
            gen_pw_linear_combine_data(args)
        if 'pw_wave_const' in args.data_type:
            gen_pw_wave_const_data(args)
        if 'pw_wave_linear_combine' in args.data_type:
            gen_pw_wave_linear_combine_data(args)
        if 'pw_wave_multiply_linear_reg' in args.data_type:
            gen_pw_wave_multiply_linear_reg_data(args)
        if 'pw_wave_add_linear_reg' in args.data_type:
            gen_pw_wave_add_linear_reg_data(args)
        if 'pw_wave_t_shift' in args.data_type:
            gen_pw_wave_t_shift_data(args)
        if 'linear_reg_cluster' in args.data_type:
            gen_linear_reg_cluster_data(args)
        if 'power_cluster' in args.data_type:
            gen_power_cluster_data(args)
        if 't_shift_cluster' in args.data_type:
            gen_t_shift_cluster_data(args)
