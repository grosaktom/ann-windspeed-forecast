import numpy
import matplotlib
import pandas
from sklearn.preprocessing import MinMaxScaler
from pandas import datetime
from pathlib import Path
numpy.random.seed(7)


def date_parser(x):
	return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

def date_parser_wrf(x):
	return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

def transpose_wrf_variables(df, variables):
    df_transposed = None
    for num in variables:
        df_filtered = df.loc[df['id_var'] == num]
        df_filtered.columns = [str(col) + '_' + variables[num] for col in df_filtered.columns]
        if (df_transposed is None):
            df_transposed = df_filtered.copy()
            df_transposed = df_transposed.rename(columns = {'datum_'+variables[num]:'datum'})
        else:
            df_transposed = pandas.merge(df_transposed.set_index("datum", drop=False), df_filtered, how='inner', left_on='datum', right_on='datum_'+variables[num])

    df_transposed.reset_index()
    df_transposed.set_index('datum', inplace=True)
    df_transposed = df_transposed.drop('id_var_wind_speed', 1)
    return df_transposed


def read_measurements(measurement_file = ''):
    if ( measurement_file == '' ):
        raise Exception('You need to supply a forecast file path!')
    file = Path(measurement_file)
    if ( file.is_file() == False ):
        raise Exception('File does not exist!')

    df_measurements = pandas.read_csv( measurement_file,
                          delimiter=',' ,
                          parse_dates=['datetime'],
                          index_col='datetime',
                          squeeze=True,
                          date_parser=date_parser ,
                          usecols=['datetime','wind_avg','wind_min','wind_max','wind_direction','temperature' ],
                          engine='python' )

    return df_measurements

def read_wrf( forecast_file = '' ):
    if ( forecast_file == '' ):
        raise Exception('You need to supply a forecast file path!')
    file = Path(forecast_file)
    if ( file.is_file() == False ):
        raise Exception('File does not exist!')

    df_wrf = pandas.read_csv(   forecast_file,
                            delimiter=',',
                            parse_dates=['datum'],
                            index_col=False,
                            squeeze=True,
                            date_parser=date_parser_wrf,
                            usecols=['datum', 'id_var', 'h0', 'h24', 'h48', 'h72'],
                            engine='python',
                            skipfooter=3)

    return df_wrf

def normalize_dataset(df, input_variables, output_variables):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = df.copy()

    df_scaled[input_variables] = scaler.fit_transform(df[input_variables])
    df_scaled[output_variables] = scaler.fit_transform(df_scaled[output_variables])

    return df_scaled

def split_dataset(merged_df_scaled, input_variables, output_variables, ratio):
    merged_df_length = len(merged_df_scaled)
    training_length = int(ratio * merged_df_length)

    training_df = merged_df_scaled[0:training_length]
    test_df = merged_df_scaled[training_length + 1:merged_df_length]

    training_x = numpy.array(training_df[input_variables])
    training_y = numpy.array(training_df[output_variables])

    test_x = numpy.array(test_df[input_variables])
    test_y = numpy.array(test_df[output_variables])

    training_x = numpy.reshape(training_x, (training_x.shape[0], 1, training_x.shape[1]))
    test_x = numpy.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

    return training_x,training_y,test_x,test_y


def shift_measurements(df_measurement_real, days):

    df_measurement_shifted = df_measurement_real.copy()
    df_measurement_shifted.index = pandas.to_datetime(df_measurement_shifted.index)
    df_measurement_shifted = df_measurement_shifted.shift(days, freq='D')
    df_measurement_shifted = df_measurement_shifted.rename(
        columns={'wind_avg': 'wind_avg_in24h', 'wind_max': 'wind_max_in24h', 'wind_min': 'wind_min_in24h', 'wind_direction': 'wind_direction_in24h',
                 'temperature': 'temperature_in24h'})
    return df_measurement_shifted



def merge_datasets(wrf_file, measurement_file):

    df_measurement_real = read_measurements(measurement_file)
    df_measurement_shifted = shift_measurements(df_measurement_real, -3)
    df_measurement = pandas.merge(df_measurement_real, df_measurement_shifted, how='inner', left_index=True, right_index=True)


    df_wrf = read_wrf( wrf_file )

    variables_to_transpond = {
                    1: 'wind_speed',
                    2: 'wind_dir',
                    3: 'temperature',
                 }

    df_transposed = transpose_wrf_variables(df_wrf, variables_to_transpond)
    df_transposed.index = pandas.to_datetime(df_transposed.index, unit='s')
    df_upsampled = df_transposed.resample('10min')
    df_wrf_interpolated = df_upsampled.interpolate(method='linear')

    merged_df = pandas.merge(df_wrf_interpolated, df_measurement, how='inner', left_index=True, right_index=True)
    merged_df.dropna( inplace=True, subset=['wind_avg', 'wind_min', 'wind_max', 'wind_direction','temperature','wind_avg_in24h','wind_direction_in24h','temperature_in24h','h24_wind_speed','h24_wind_dir','h24_temperature','h0_wind_speed','h0_wind_dir','h0_temperature' ])

    return merged_df


