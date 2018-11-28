import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import GaussianNoise
from keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.signal import savgol_filter

import functions_single



#--------------------------------------------------------------------------------------------
#           Create dataset
#--------------------------------------------------------------------------------------------


wrf_file = 'data/station_data_1033_2016010100-20180823-rhodos-wrf9.csv'
measurement_file = 'data/station_data_1033_2016010100-20181018-rhodos-mereni.csv'

#wrf_file = 'data/station_data_632_2016010100-20181124-garda-wrf9.csv'
#measurement_file = 'data/station_data_632_2016010100-20181124-garda-mereni.csv'

#wrf_file = 'data/station-data-55-zloncice-wrf.csv'
#measurement_file = 'data/station-data-55-zloncice-measurement.csv'

#wrf_file = 'data/station_data_525_2016010100-20181121-hajovka-wrf9.csv'
#measurement_file = 'data/station_data_525_2016010100-20181121-hajovka-measurements.csv'

epochs = 10
#epochs = 20
#epochs = 30

merged_df = functions_single.merge_datasets(wrf_file, measurement_file)


#--------------------------------------------------------------------------------------------
#           Pick variables and normalize dataset
#--------------------------------------------------------------------------------------------


# input_variables = ['wind_avg','wind_direction','temperature','h0_wind_speed','h0_wind_dir','h0_temperature', 'h0_high_cloud_cover', 'h0_middle_cloud_cover', 'h0_low_cloud_cover', 'h0_relative_humidity']
#input_variables = ['h24_wind_speed','h24_wind_dir']
input_variables = ['h24_wind_speed','h24_wind_dir','h24_temperature','wind_direction','temperature']
output_variables = ['wind_avg_in24h']

merged_df_scaled = functions_single.normalize_dataset(merged_df, input_variables=input_variables, output_variables=output_variables)
ratio = 0.6
training_x, training_y, test_x, test_y = functions_single.split_dataset(merged_df_scaled, input_variables=input_variables, output_variables=output_variables, ratio=ratio)


model = Sequential()
model.add(Dense(100, input_shape=(1, len(input_variables) )))
model.add(GaussianNoise(0.1))
model.add(Dense(500 ))
model.add(Dropout(0.5))
model.add(Dense(500 ))
model.add(Dense(100 ))
model.add(Flatten())
model.add(Dense(len(output_variables), activation='sigmoid'))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(training_x, training_y, epochs=epochs, batch_size=20, verbose=2)

merged_df_length = len(merged_df)
training_length = int(ratio * merged_df_length)

train_predict = model.predict(training_x)
test_predict = model.predict(test_x, batch_size=1)

train_predict_plot = train_predict[:,0]
train_predict_plot = numpy.reshape(train_predict_plot, ( training_length, 1))



test_predict_plot = numpy.empty(merged_df_length)
test_predict_plot[:] = numpy.nan
test_predict_plot[len(train_predict_plot)+1:merged_df_length] = test_predict[:,0]


#test_predict_smooth = savgol_filter(test_predict, 7, 3) # window size 51, polynomial order 3
#test_predict_plot_smooth = numpy.empty(merged_df_length)
#test_predict_plot_smooth[:] = numpy.nan
#test_predict_plot_smooth[len(train_predict_plot)+1:merged_df_length] = test_predict_smooth[:,0]


scaled_wind_avg = numpy.array(merged_df_scaled['wind_avg_in24h'])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(training_y[:,0], train_predict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_y[:,0], test_predict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# calculate mean absolute error
trainScore = mean_absolute_error(training_y[:,0], train_predict[:,0])
print('Train Score: %.2f MAE' % (trainScore))
testScore = mean_absolute_error(test_y[:,0], test_predict[:,0])
print('Test Score: %.2f MAE' % (testScore))


scaled_wrf_wind_speed = numpy.array(merged_df_scaled['h24_wind_speed'])


plt.show()

plt.plot(scaled_wind_avg, "b")
#plt.plot(test_predict_plot_smooth, "y--")
plt.plot(train_predict_plot)
plt.plot(test_predict_plot, "r")
plt.show()




