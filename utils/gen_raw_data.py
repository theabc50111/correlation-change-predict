import argparse
import logging
from itertools import cycle
from pathlib import Path
from pprint import pformat

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from ruptures.utils import draw_bkps

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
        tuple: signal of shape (n_samples, 1), list of breakpoints
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


def pw_adapt_cov_mean_linear(n_samples=200, n_features=1, n_bkps=3, noise_std=0, cov_loc=50, seed=None):
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
        save_dir = current_dir/f'../dataset/synthetic/dim{dim}'
        save_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_dir/f'pw_constant-bkps{n_bkps}-noise_std{sigma}.csv')

    return signal, bkps


def gen_pw_linear_data(args):
    """
    Generate piecewise linear data.
    ref:
        - https://centre-borelli.github.io/ruptures-docs/code-reference/datasets/pw_linear-reference/#ruptures.datasets.pw_linear.pw_linear
        - https://centre-borelli.github.io/ruptures-docs/user-guide/datasets/pw_linear/
    """
    n, dim = args.time_len, args.dim  # time_length(number of samples), number of variables(dimension)
    n_bkps, sigma = args.n_bkps, args.noise_std  # number of change points, noise standart deviation
    signal, bkps = pw_adapt_cov_mean_linear(n, dim, n_bkps, noise_std=sigma, seed=0)
    dates = pd.to_datetime(range(n), unit='D', origin=pd.Timestamp('now'))  # create a DatetimeIndex with interval of one day
    var_names = ['var_linear_sum']+[f'var_{i}' for i in range(dim)]
    df = pd.DataFrame(signal, index=dates, columns=var_names)
    df.index.name = "Date"
    logger.info(f"Generated piecewise linear data with shape {df.shape} and {n_bkps} change points.")
    logger.info(f"piecewiase linear data[:5, :5]:\n{df.iloc[:5, :5]}")
    if args.save_data:
        save_dir = current_dir/f'../dataset/synthetic/dim{dim}'
        save_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_dir/f'pw_linear-bkps{n_bkps}-noise_std{sigma}.csv')

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
        save_dir = current_dir/f'../dataset/synthetic/dim{dim}'
        save_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_dir/f'pw_wave_const-bkps{n_bkps}-noise_std{sigma}.csv')

    return signal, bkps


def gen_pw_wave_linear_data(args):
    """
    Generate piecewise wave * linear data.
    """
    n, dim = args.time_len, args.dim  # time_length(number of samples), number of variables(dimension)
    n_bkps, sigma = args.n_bkps, args.noise_std  # number of change points, noise standart deviation
    wave_signal, bkps = pw_rand_f1_f2_wavy(n, n_bkps, noise_std=0)
    linear_signal, bkps = pw_adapt_cov_mean_linear(n, dim, n_bkps, noise_std=sigma, seed=0)
    signal = wave_signal.reshape(n, 1)*linear_signal
    dates = pd.to_datetime(range(n), unit='D', origin=pd.Timestamp('now'))  # create a DatetimeIndex with interval of one day
    var_names = ['var_linear_sum']+[f'var_{i}' for i in range(dim)]
    df = pd.DataFrame(signal, index=dates, columns=var_names)
    df.index.name = "Date"
    logger.info(f"Generated piecewise wave_linear data with shape {df.shape} and {n_bkps} change points.")
    logger.info(f"piecewiase wave_linear data[:5, :5]:\n{df.iloc[:5, :5]}")
    if args.save_data:
        save_dir = current_dir/f'../dataset/synthetic/dim{dim}'
        save_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_dir/f'pw_wave_linear-bkps{n_bkps}-noise_std{sigma}.csv')

    return signal, bkps


def gen_multi_cluster_pw_wave_const_data(args):
    """
    Generate multiple cluseter piecewise wave * const data.
    Construct the clusters by multply wave and linear data, each cluster has different wave.
    The variables of linear data of cluster is consist of response variable and its covariates, the response variable is a linear combination of the covariates.
    """
    n, dim = args.time_len, args.dim  # time_length(number of samples), number of variables(dimension)
    n_bkps, sigma = args.n_bkps, args.noise_std  # number of change points, noise standart deviation
    n_clusters = args.n_clusters  # number of clusters
    signal = np.zeros((n, n_clusters*dim))
    for sub, cluster_idx in zip(np.split(signal, n_clusters, axis=1), range(n_clusters)):
        cluster_signal, bkps = gen_pw_wave_const_data(args, seed=cluster_idx)
        sub += cluster_signal
    dates = pd.to_datetime(range(n), unit='D', origin=pd.Timestamp('now'))  # create a DatetimeIndex with interval of one day
    var_names = [f'cluster_{cluster_idx}_var_{i}' for cluster_idx in range(n_clusters) for i in range(dim)]
    df = pd.DataFrame(signal, index=dates, columns=var_names)
    df.index.name = "Date"
    logger.info(f"Generated {n_clusters}_clusters piecewise wave_const data with shape {df.shape} and {n_bkps} change points.")
    logger.info(f"piecewiase wave_linear data[:5, :5]:\n{df.iloc[:5, :5]}")
    if args.save_data:
        save_dir = current_dir/f'../dataset/synthetic/dim{n_clusters*dim}'
        save_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_dir/f'cluster_{n_clusters}-pw_wave_const-bkps{n_bkps}-noise_std{sigma}.csv')


def gen_pw_2_layers_wave_linear_data(args):
    """
    Generate 2 layer piecewise wave * linear data.
    Construct the clusters for sub layer, then use the linear_sum column(response variable) of each cluster in sub layer to construct the second layer.
    The variables of cluster in sub layer is consist of response variable and its covariates, the response variable is a linear combination of the covariates.
    """
    n, dim = args.time_len, args.dim  # time_length(number of samples), number of variables(dimension)
    n_bkps, sigma = args.n_bkps, args.noise_std  # number of change points, noise standart deviation
    wave_signal, bkps = pw_rand_f1_f2_wavy(n, n_bkps, noise_std=0)
    linear_signal, bkps = pw_adapt_cov_mean_linear(n, dim, n_bkps, noise_std=sigma, seed=0)
    signal = wave_signal.reshape(n, 1)*linear_signal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate raw data.')
    parser.add_argument('--data_type', type=str, default=['pw_constant'], nargs='+',
                        choices=['pw_constant', 'pw_linear', 'pw_wave_const', 'pw_wave_linear',
                                 'multi_cluster_pw_wave_const'],
                        help='Type of data to generate. (default: pw_constant)')
    parser.add_argument('--time_len', type=int, default=2600, help='Input time length. (default: 2600)')
    parser.add_argument('--dim', type=int, default=70, help='Input dimension(number of variable). (default: 70)')
    parser.add_argument('--noise_std', type=int, default=2, help='Input noise standard deviation. (default: 2))')
    parser.add_argument('--n_bkps', type=int, default=0, help='Input number of change points. (default: 0)')
    parser.add_argument('--n_clusters', type=int, default=0, help='Input number of clusters. (default: 0)')
    parser.add_argument("--save_data", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="input --save_data to save raw data")
    args = parser.parse_args()
    assert not ("multi_cluster_pw_wave_const" in args.data_type) ^ (args.n_clusters > 0), "`n_clusters` should be set when `data_type` is 'multi_cluster_pw_wave_const'"
    logger.info(pformat(vars(args), indent=1, width=100, compact=True))

    if 'pw_constant' in args.data_type:
        gen_pw_constant_data(args)
    if 'pw_linear' in args.data_type:
        gen_pw_linear_data(args)
    if 'pw_wave_const' in args.data_type:
        gen_pw_wave_const_data(args)
    if 'pw_wave_linear' in args.data_type:
        gen_pw_wave_linear_data(args)
    if 'multi_cluster_pw_wave_const' in args.data_type:
        gen_multi_cluster_pw_wave_const_data(args)
