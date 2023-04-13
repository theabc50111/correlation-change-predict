import logging

import numpy as np
import pandas as pd
from pmdarima.arima import ARIMA, auto_arima
from tqdm import tqdm


def arima_err_logger_init(log_file_path: "pathlib.PosixPath"):
    global err_logger
    formatter = logging.Formatter("""
==============================================
%(asctime)s - %(name)s - %(levelname)s
%(message)s
""")
    err_log_handler = logging.FileHandler(filename=log_file_path/"arima_train_err_log.txt", mode='a')
    err_log_handler.setFormatter(formatter)
    err_logger = logging.getLogger("arima_train_err")
    err_logger.addHandler(err_log_handler)


def arima_model(dataset: "pd.DataFrame", arima_result_path_basis: "pathlib.PosixPath", data_split_setting: str = "", save_file: bool = False) -> ("pd.DataFrame", "pd.DataFrame", "pd.DataFrame"):
    """
    Note: before operate arima_model(), need to operate arima_err_logger_init() first.
    """
    model_110 = ARIMA(order=(1, 1, 0), out_of_sample_size=0, mle_regression=True, suppress_warnings=True)
    model_011 = ARIMA(order=(0, 1, 1), out_of_sample_size=0, mle_regression=True, suppress_warnings=True)
    model_111 = ARIMA(order=(1, 1, 1), out_of_sample_size=0, mle_regression=True, suppress_warnings=True)
    model_211 = ARIMA(order=(2, 1, 1), out_of_sample_size=0, mle_regression=True, suppress_warnings=True)
    model_210 = ARIMA(order=(2, 1, 0), out_of_sample_size=0, mle_regression=True, suppress_warnings=True)
    #model_330 = ARIMA(order=(3, 3, 0), out_of_sample_size=0, mle_regression=True, suppress_warnings=True)

    #model_dict = {"model_110": model_110, "model_011": model_011, "model_111": model_111, "model_211": model_211, "model_210": model_210, "model_330": model_330}
    model_dict = {"model_110": model_110, "model_011": model_011, "model_111": model_111, "model_211": model_211, "model_210": model_210}
    tested_models = []
    arima_model = None
    find_arima_model = False
    arima_attr_list = ["aic", "arparams", "aroots", "maparams", "maroots", "params", "pvalues"]
    arima_output_list = []
    arima_resid_list = []
    arima_model_info_list = []
    for corr_pair, corr_series in tqdm(dataset.iterrows()):
        while not find_arima_model:
            try:
                for model_key in model_dict:
                    if model_key not in tested_models:
                        test_model = model_dict[model_key].fit(corr_series[:-1]) # only use first 20 corrletaion coefficient to fit ARIMA model
                        if arima_model is None:
                            arima_model = test_model
                            arima_model_name = model_key
                        elif arima_model.aic() <= test_model.aic():
                            pass
                        else:
                            arima_model = test_model
                            arima_model_name = model_key
                    tested_models.append(model_key)
            except Exception:
                if len(model_dict)-1 != 0:
                    del model_dict[model_key]
                else:
                    err_logger.error(f"fatal error, {corr_pair} doesn't have appropriate arima model\n", exc_info=True)
                    break
            else:
                #model_dict = {"model_110": model_110, "model_011": model_011, "model_111": model_111, "model_211": model_211, "model_210": model_210, "model_330": model_330}
                model_dict = {"model_110": model_110, "model_011": model_011, "model_111": model_111, "model_211": model_211, "model_210": model_210}
                tested_models.clear()
                find_arima_model = True
        try:
            arima_pred = list(arima_model.predict(n_periods=1))
        except Exception:
            err_logger.error(f"{corr_pair} in {data_split_setting} be predicted by {arima_model_name}(its aic:{arima_model.aic()}) getting error:\n", exc_info=True)
            dataset = dataset.drop(index=corr_pair)
        else:
            arima_pred_in_sample = list(arima_model.predict_in_sample())
            arima_pred_in_sample = [np.mean(arima_pred_in_sample[1:])] + arima_pred_in_sample[1:]
            arima_output = arima_pred_in_sample + arima_pred
            arima_output = np.clip(np.array(arima_output), -1, 1)
            arima_output_list.append(arima_output)

            arima_resid = pd.Series(np.array(corr_series) - arima_output)
            arima_resid_list.append(np.array(arima_resid))
            arima_infos = [corr_pair, arima_model_name]
            for attr in arima_attr_list:
                try:
                    val = getattr(arima_model, attr)()
                except AttributeError:
                    arima_infos.append(None)
                else:
                    arima_infos.append(val)
            else:
                arima_model_info_list.append(arima_infos)
        finally:
            find_arima_model = False


    arima_model_info_df = pd.DataFrame(arima_model_info_list, dtype="float32", columns=["items", "arima_model", "arima_aic", "arima_pvalues", "arima_params", "arima_arparams", "arima_aroots", "arima_maparams", "arima_maroots"]).set_index("items")
    arima_output_df = pd.DataFrame(arima_output_list, dtype="float32", index=dataset.index)
    arima_output_df.index.name = "items"
    arima_resid_df = pd.DataFrame(arima_resid_list, dtype="float32", index=dataset.index)
    arima_resid_df.index.name = "items"

    if save_file:
        arima_model_info_df.to_csv(arima_result_path_basis.parent/(str(arima_result_path_basis.stem) + f'-arima_model_info{data_split_setting}.csv'))
        arima_output_df.to_csv(arima_result_path_basis.parent/(str(arima_result_path_basis.stem) + f'-arima_output{data_split_setting}.csv'))
        arima_resid_df.to_csv(arima_result_path_basis.parent/(str(arima_result_path_basis.stem) + f'-arima_resid{data_split_setting}.csv'))

    return arima_output_df, arima_resid_df, arima_model_info_df
