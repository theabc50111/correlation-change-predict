#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# # Create Portfolio
# 
# out of 505 companies, 150 firms are randomly selected for the portfolio.


df = pd.read_csv("../../stock08_price.csv")
universe = list(df.columns.values[1:])
universe.remove("SP500")
# print(universe)


# train data
portfolio_train = ['CELG', 'PXD', 'WAT', 'LH', 'AMGN', 'AOS', 'EFX', 'CRM', 'NEM', 'JNPR', 'LB', 'CTAS', 'MAT', 'MDLZ', 'VLO', 'APH', 'ADM', 'MLM', 'BK', 'NOV', 'BDX', 'RRC', 'IVZ', 'ED', 'SBUX', 'GRMN', 'CI', 'ZION', 'COO', 'TIF', 'RHT', 'FDX', 'LLL', 'GLW', 'GPN', 'IPGP', 'GPC', 'HPQ', 'ADI', 'AMG', 'MTB', 'YUM', 'SYK', 'KMX', 'AME', 'AAP', 'DAL', 'A', 'MON', 'BRK', 'BMY', 'KMB', 'JPM', 'CCI', 'AET', 'DLTR', 'MGM', 'FL', 'HD', 'CLX', 'OKE', 'UPS', 'WMB', 'IFF', 'CMS', 'ARNC', 'VIAB', 'MMC', 'REG', 'ES', 'ITW', 'NDAQ', 'AIZ', 'VRTX', 'CTL', 'QCOM', 'MSI', 'NKTR', 'AMAT', 'BWA', 'ESRX', 'TXT', 'EXR', 'VNO', 'BBT', 'WDC', 'UAL', 'PVH', 'NOC', 'PCAR', 'NSC', 'UAA', 'FFIV', 'PHM', 'LUV', 'HUM', 'SPG', 'SJM', 'ABT', 'CMG', 'ALK', 'ULTA', 'TMK', 'TAP', 'SCG', 'CAT', 'TMO', 'AES', 'MRK', 'RMD', 'MKC', 'WU', 'ACN', 'HIG', 'TEL', 'DE', 'ATVI', 'O', 'UNM', 'VMC', 'ETFC', 'CMA', 'NRG', 'RHI', 'RE', 'FMC', 'MU', 'CB', 'LNT', 'GE', 'CBS', 'ALGN', 'SNA', 'LLY', 'LEN', 'MAA', 'OMC', 'F', 'APA', 'CDNS', 'SLG', 'HP', 'XLNX', 'SHW', 'AFL', 'STT', 'PAYX', 'AIG', 'FOX', 'MA']
# all data
portfolio_all = universe
# all data - train data
portfolio_other = [p for p in universe if p not in portfolio_train]
print(len(portfolio_train), len(portfolio_all), len(portfolio_other))


# # Prepare the Data


def rolling_corr(item1: str, item2: str) -> "pd.series":
    # import data
    stock_price_df = pd.read_csv("../../stock08_price.csv")
    pd.to_datetime(stock_price_df['Date'], format='%Y-%m-%d')
    stock_price_df = stock_price_df.set_index(pd.DatetimeIndex(stock_price_df['Date']))

    # calculate
    df_pair = pd.concat([stock_price_df[item1], stock_price_df[item2]], axis=1)
    df_corr = df_pair[item1].rolling(window=100).corr(df_pair[item2])
    return df_corr


def gen_data(portfolio: list, file_name: str = "", save_file: bool = False) -> "pd.DataFrame":
    index_list = []
    for _ in range(100):
        indices = []
        for k in range(_, 2420, 100):
            indices.append(k)
        index_list.append(indices)

    data_matrix = []
    count = 0
    for i in range(len(portfolio)):
        for j in range(len(portfolio)-1-i):
            a = portfolio[i]
            b = portfolio[len(portfolio)-1-j]
            corr_series = rolling_corr(a, b)[99:]
            for _ in range(100):
                corr_strided = list(corr_series[index_list[_]][:24]).copy()
                data_matrix.append(corr_strided)
                count += 1
                if count % 100000 == 0:
                    print(str(count)+' items preprocessed')

    data_matrix = np.transpose(data_matrix)
    data_dictionary = {}
    for i in range(len(data_matrix)):
        data_dictionary[str(i)] = data_matrix[i]
    data_df = pd.DataFrame(data_dictionary)
    if save_file:
        data_df.to_csv(f'./correlation_record/{file_name}', index=False)

    return data_df


def gen_corr_series(data_df=None, file_name="", from_file=False):
    if from_file:
        data_df = pd.read_csv(f'./correlation_record/{file_name}')
        data_df = data_df.loc[:, ~data_df.columns.str.contains('^Unnamed')]
    ind_range = int(len(data_df)/20)
    num_list = []
    for i in range(24):
        num_list.append(str(i))
    data_df = data_df[num_list].copy()
    data_df = np.transpose(data_df)
    indices = [20*k for k in range(ind_range)]
    data_df = pd.DataFrame(data_df[indices])
    return data_df.values.reshape(-1,)


if __name__ == "__main__":
    gen_from_file = True

    if gen_from_file:
        train_corr_series = gen_corr_series(None, "train_dataset.csv", True)
        all_corr_series = gen_corr_series(None, "445_dataset.csv", True)
        other_corr_series = gen_corr_series(None, "295_dataset.csv", True)
    else:
        train_data_df = gen_data(portfolio_train)
        all_data_df = gen_data(portfolio_all)
        other_data_df = gen_data(portfolio_other)
        train_corr_series = gen_corr_series(train_data_df)
        all_corr_series = gen_corr_series(all_data_df)
        other_corr_series = gen_corr_series(other_data_df)
