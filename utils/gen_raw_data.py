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

# creation of data
def gen_pw_constant_data(args):
    """
    Generate piecewise constant data.
    ref: https://centre-borelli.github.io/ruptures-docs/code-reference/datasets/pw_constant-reference/
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate raw data.')
    parser.add_argument('--time_len', type=int, default=2600, help='Input time length.')
    parser.add_argument('--dim', type=int, default=70, help='Input dimension(number of variable).')
    parser.add_argument('--noise_std', type=int, default=10, help='Input noise standard deviation.')
    parser.add_argument('--n_bkps', type=int, default=0, help='Input number of change points.')
    parser.add_argument("--save_data", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="input --save_data to save raw data")
    args = parser.parse_args()
    logger.info(pformat(vars(args), indent=1, width=100, compact=True))
    gen_pw_constant_data(args)

