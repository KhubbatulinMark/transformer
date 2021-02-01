import os
import glob
import pandas as pd


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


def get_data_from_csv(file_dir, ext, predfile=False):
    all_files = glob.glob(file_dir + "/*." + ext)
    data = pd.DataFrame(columns=['H2', 'CO', 'C2H4', 'C2H2', 'file_name', 'predict_dt'])
    print(predfile)
    if predfile:
        print(1)
        datafrompredfile = pd.read_csv(predfile).to_dict()
    return datafrompredfile
