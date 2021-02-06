import os
import glob
import pandas as pd
import warnings
from itertools import product
from transformator import get_accepted_maximum_value
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pmdarima as pm


def get_filename_with_status(filename_list):
    filename_with_status = {}
    for filename in filename_list:
        basename = os.path.basename(filename)
        for status in ['LED', 'norm', 'LTO', 'PD']:
            if status in basename:
                filename_with_status[filename] = status
                break
    return filename_with_status


def get_all_data(file_dir, ext):
    all_files = glob.glob(file_dir + "/*." + ext)
    data = pd.DataFrame(columns=['H2', 'CO', 'C2H4', 'C2H2', 'status'])
    file_status = get_filename_with_status(all_files)
    for key, value in file_status.items():
        df = pd.read_table(key, encoding='cp1251', header=None, skiprows=2)
        datetime_list = pd.date_range("2020-01-01", periods=len(df), freq='12H')
        for i in range(len(df)):
            gas_concentration = df.iloc[i][0].split('  ')[2:10:2]
            data = data.append(
                {'datetime': datetime_list[i],
                 'H2': gas_concentration[0],
                 'CO': gas_concentration[1],
                 'C2H4': gas_concentration[2],
                 'C2H2': gas_concentration[3],
                 'status': value}, ignore_index=True)
    data = data.astype({'H2': 'float128', 'CO': 'float128', 'C2H4': 'float128', 'C2H2': 'float128'})
    return data


def get_data_from_csv(file_dir, ext, predfile=None):

    if predfile:
        parse = []
        datafrompredfile = pd.read_csv(predfile).values
        datafrompredfile = {i[0]:i[1] for i in datafrompredfile}
        for i,key in enumerate(datafrompredfile):
            file = glob.glob(file_dir + "/"+key)
            datafromfile=pd.read_csv(file[0])
            datetime_list = pd.date_range("2020-01-01", periods=len(datafromfile), freq='12H')
            datafromfile.reset_index(drop=True,inplace=True)
            datafromfile = datafromfile.set_index(datetime_list)
            parse.append((key,datafrompredfile[key],datafromfile))
    else:
        parse = []
        files = glob.glob(file_dir + "/*." + ext)
        for i, file in enumerate(files):
            datafromfile = pd.read_csv(file)
            datetime_list = pd.date_range("2020-01-01", periods=len(datafromfile), freq='12H')
            datafromfile.reset_index(drop=True, inplace=True)
            datafromfile = datafromfile.set_index(datetime_list)
            parse.append((Path(file).name, None, datafromfile))
    return parse
def find_best_arima_model_for_gas(data):
    file = data
    data_normal = file[2]
    model_res = {}
    for i, gas in enumerate(['H2', 'CO', 'C2H4', 'C2H2']):
        # print('----------------------------------------------------\nGAS %s' % gas)
        df = pd.DataFrame(columns=['aic', 'param'])
        p = q = range(0, 2)
        d =[1,2]
        pdq = list(product(p, d, q))
        warnings.filterwarnings("ignore")
        aics = []
        params = []
        # all=len(pdq)*len(seasonal_pdq)
        # pbar = tqdm(total=all)
        for param in pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(data_normal[gas],
                                                order=param,

                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()

                # print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
                aics.append(results.aic)
                params.append(param)
                # pbar.update(1)
            except BaseException as e:
                continue
        df = pd.DataFrame({'aic': aics, 'param': params})
        minaic_param = df[df.aic == df.aic.min()].iloc[[0]]
        # print(minaic_param)
        # print(minaic_param.param.values, minaic_param.param_seasonal.values)
        mod = sm.tsa.statespace.SARIMAX(data_normal[gas],
                                        order=minaic_param.param.values[0],
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        model_res[gas] = mod.fit()

        # print(model_res[gas].summary().tables[1])
        # pbar.close()
    return model_res

def plot_predict(data,model_res):
    file=data
    data_normal = file[2]
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    for i, gas in enumerate(['H2', 'CO', 'C2H4', 'C2H2']):
        print('----------------------------------------------------\nGAS %s' % gas)
        plt.figure()
        pred = model_res[gas].get_prediction(start=pd.to_datetime('2020-04-01'), dynamic=False)
        pred_ci = pred.conf_int()
        ax = data_normal[gas]['2020-04-01':].plot(label='observed')
        pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 4))
        ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Conc')
        plt.legend()
        y_forecasted = pred.predicted_mean
        y_truth = data_normal[gas]['2020-04-01':]
        mse = ((y_forecasted - y_truth) ** 2).mean()
        print('The Mean Squared Error is {}'.format(round(mse, 10)))
        print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 10)))
        plt.figure()
        pred_val = model_res[gas].forecast(steps=file[1])
        pred_uc = model_res[gas].get_forecast(steps=file[1])
        pred_ci = pred_uc.conf_int()
        ax = y_truth.plot(label='observed', figsize=(14, 4))
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
        datetime_pred = pd.date_range("2020-01-01", periods=file[1], freq='12H')

        accepted_level = get_accepted_maximum_value(gas, 0, '35kW')[0]
        maximum_level = get_accepted_maximum_value(gas, 0, '35kW')[1]
        max_axhspan_level = max(data_normal[gas].max(), maximum_level, pred_val.max()) * 1.05

        max_text_level = (max_axhspan_level - maximum_level) / 2 + maximum_level
        accepted_text_level = (maximum_level - accepted_level) / 2 + accepted_level
        normal_text_level = accepted_level / 2
        text_egle = round(len(data_normal.index) * 0.90)
        # Графики
        plt.vlines(datetime_pred[-1], 0, max_axhspan_level,
               color='r',
               linewidth=2,
               linestyle='--')
        # Зоны
        plt.axhspan(0, accepted_level, facecolor='1', color='green', alpha=0.3)
        plt.axhspan(accepted_level, maximum_level, facecolor='1', color='yellow', alpha=0.3)
        plt.axhspan(maximum_level, max_axhspan_level, facecolor='1', color='red', alpha=0.3)
        #         Текст
        plt.text(data_normal.index[text_egle], max_text_level, "Предотказное состояние", fontsize=12, color='black',
             bbox=props)
        plt.text(data_normal.index[text_egle], accepted_text_level, "Развитие дефекта", fontsize=12, color='black',
             bbox=props)
        plt.text(data_normal.index[text_egle], normal_text_level, "Нормальное состояние", fontsize=12, color='black',
             bbox=props)
        #     ax.fill_between(pred_ci.index,
        #                     pred_ci.iloc[:, 0],
        #                     pred_ci.iloc[:, 1], color='k', alpha=.25)
def find_best_autoarima_model_for_gas(data):
    file = data
    data_normal = file[2]
    model_res = {}
    for i, gas in enumerate(['H2', 'CO', 'C2H4', 'C2H2']):
        print('----------------------------------------------------\nGAS %s' % gas)
        df = pd.DataFrame(columns=['aic', 'param', 'param_seasonal'])
        d = p = q = range(0, 3)
        pdq = list(product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 1) for x in pdq]
        warnings.filterwarnings("ignore")
        aics = []
        params = []
        param_seasonals = []
        all=len(pdq)*len(seasonal_pdq)
        pbar = tqdm(total=all)
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(data_normal[gas],
                                                    order=param,

                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)

                    results = mod.fit()

                    # print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
                    aics.append(results.aic)
                    params.append(param)
                    param_seasonals.append(param_seasonal)
                    pbar.update(1)
                except BaseException as e:
                    continue
        df = pd.DataFrame({'aic': aics, 'param': params, 'param_seasonal': param_seasonals})
        minaic_param = df[df.aic == df.aic.min()].iloc[[0]]
        # print(minaic_param)
        # print(minaic_param.param.values, minaic_param.param_seasonal.values)
        mod = sm.tsa.statespace.SARIMAX(data_normal[gas],
                                        order=minaic_param.param.values[0],
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)

        model_res[gas] = mod.fit()

        # print(model_res[gas].summary().tables[1])
        pbar.close()
    return model_res


def auto_arima(data):
    file = data
    data_normal = file[2]
    model_res = {}
    for i, gas in enumerate(['H2', 'CO', 'C2H4', 'C2H2']):
        stepwise_fit = pm.auto_arima(data_normal[gas], start_p=1, start_q=1, max_p=2, max_q=1, max_d=2,
                                     seasonal=False, trace=False,
                                     error_action='ignore',
                                     suppress_warnings=True,
                                     stepwise=True, n_jobs=8)
        model_res[gas] = stepwise_fit
    return model_res


def plot_predict_auto_arima(data, model_res):
    file = data
    data_normal = file[2]
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    for i, gas in enumerate(['H2', 'CO', 'C2H4', 'C2H2']):
        pred = model_res[gas].predict(n_periods=file[1])
        datetime_pred = pd.date_range("2020-01-01", periods=len(data_normal[gas]) + file[1], freq='12H')
        plt.figure()
        forecasts = pd.DataFrame(pred, index=pd.date_range(datetime_pred[len(data_normal[gas])], datetime_pred[-1],
                                                           freq='12H'))

        # set up the plot

        # add the lines
        plt.plot(data_normal[gas], color='b', label='Observed')
        plt.plot(forecasts, color='r', label='Forecast')

        accepted_level = get_accepted_maximum_value(gas, 0, '220kW')[0]
        maximum_level = get_accepted_maximum_value(gas, 0, '220kW')[1]
        max_axhspan_level = max(data_normal[gas].max(), maximum_level, pred.max()) * 1.05

        max_text_level = (max_axhspan_level - maximum_level) / 2 + maximum_level
        accepted_text_level = (maximum_level - accepted_level) / 2 + accepted_level
        normal_text_level = accepted_level / 2
        text_egle = round(len(data_normal.index) * 0.90)
        # Графики
        plt.vlines(datetime_pred[-1], 0, max_axhspan_level,
                   color='r',
                   linewidth=2,
                   linestyle='--')
        # Зоны
        plt.axhspan(0, accepted_level, facecolor='1', color='green', alpha=0.3)
        plt.axhspan(accepted_level, maximum_level, facecolor='1', color='yellow', alpha=0.3)
        plt.axhspan(maximum_level, max_axhspan_level, facecolor='1', color='red', alpha=0.3)
        #         Текст
        plt.text(data_normal.index[text_egle], max_text_level, "Предотказное состояние", fontsize=12, color='black',
                 bbox=props)
        plt.text(data_normal.index[text_egle], accepted_text_level, "Развитие дефекта", fontsize=12, color='black',
                 bbox=props)
        plt.text(data_normal.index[text_egle], normal_text_level, "Нормальное состояние", fontsize=12, color='black',
                 bbox=props)


def predict_time_st(models, file):
    time_gas = []
    for i, gas in enumerate(['H2', 'CO', 'C2H4', 'C2H2']):
        accepted_level = get_accepted_maximum_value(gas, 0, '220kW')[0]
        pered_val = models[gas].predict(n_periods=1500)
        time_gas = np.argwhere(pered_val > accepted_level)
        if time_gas.size == 0:
            stepwise_fit = pm.auto_arima(file[2][gas], start_p=1, start_q=1, max_p=2, max_q=2, max_d=2,
                                         start_P=0, seasonal=False, trace=False,
                                         error_action='ignore',
                                         suppress_warnings=True,
                                         stepwise=True, n_jobs=8)
            pered_val = stepwise_fit.predict(n_periods=1500)
            time_gas = np.argwhere(pered_val > accepted_level)

    return time_gas.min()
