import argparse
import logging
from pathlib import Path
from pprint import pformat

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import ruptures as rpt

current_dir = Path(__file__).parent

logger = logging.getLogger(__name__)
logger_console = logging.StreamHandler()
logger_formatter = logging.Formatter('%(levelname)-8s [%(filename)s] %(message)s')
logger_console.setFormatter(logger_formatter)
logger.addHandler(logger_console)
logger.setLevel(logging.INFO)

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


def gen_pw_linear_data(args):
    """
    Generate piecewise linear data.
    ref:
        - https://centre-borelli.github.io/ruptures-docs/code-reference/datasets/pw_linear-reference/#ruptures.datasets.pw_linear.pw_linear
        - https://centre-borelli.github.io/ruptures-docs/user-guide/datasets/pw_linear/
    """
    n, dim = args.time_len, args.dim  # time_length(number of samples), number of variables(dimension)
    n_bkps, sigma = args.n_bkps, args.noise_std  # number of change points, noise standart deviation
    signal, bkps = rpt.pw_linear(n, dim, n_bkps, noise_std=sigma, seed=0)
    signal = signal+abs(signal.min())+1
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


def gen_pw_wave_const_data(args):
    """
    Generate piecewise wave * constant data.
    """
    n, dim = args.time_len, args.dim  # time_length(number of samples), number of variables(dimension)
    n_bkps, sigma = args.n_bkps, args.noise_std  # number of change points, noise standart deviation
    wave_signal, bkps = rpt.pw_wavy(n, n_bkps, noise_std=0, seed=0)
    const_signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma, delta=(1, 500), seed=0)
    signal = wave_signal.reshape(n, 1)*const_signal
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate raw data.')
    parser.add_argument('--time_len', type=int, default=2600, help='Input time length. (default: 2600)')
    parser.add_argument('--dim', type=int, default=70, help='Input dimension(number of variable). (default: 70)')
    parser.add_argument('--noise_std', type=int, default=2, help='Input noise standard deviation. (default: 2))')
    parser.add_argument('--n_bkps', type=int, default=0, help='Input number of change points. (default: 0)')
    parser.add_argument("--save_data", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="input --save_data to save raw data")
    args = parser.parse_args()
    logger.info(pformat(vars(args), indent=1, width=100, compact=True))
    gen_pw_constant_data(args)
    gen_pw_linear_data(args)
    gen_pw_wave_const_data(args)

