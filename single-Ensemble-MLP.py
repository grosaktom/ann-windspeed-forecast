import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import GaussianNoise
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import functions_single

#--------------------------------------------------------------------------------------------
#           Create dataset
#--------------------------------------------------------------------------------------------



#wrf_file = 'data/station_data_1033_2016010100-20180823-rhodos-wrf9.csv'
#measurement_file = 'data/station_data_1033_2016010100-20181018-rhodos-mereni.csv'

#wrf_file = 'data/station_data_632_2016010100-20181124-garda-wrf9.csv'
#measurement_file = 'data/station_data_632_2016010100-20181124-garda-mereni.csv'

#wrf_file = 'data/station-data-55-zloncice-wrf.csv'
#measurement_file = 'data/station-data-55-zloncice-measurement.csv'

wrf_file = 'data/station_data_525_2016010100-20181121-hajovka-wrf9.csv'
measurement_file = 'data/station_data_525_2016010100-20181121-hajovka-measurements.csv'

epochs = 10

merged_df = functions_single.merge_datasets(wrf_file, measurement_file)

#--------------------------------------------------------------------------------------------
#           Pick variables and normalize dataset
#--------------------------------------------------------------------------------------------


input_variables = ['wind_direction']
input_variables_mlp = ['h72_wind_speed','h72_wind_dir','h72_temperature', 'wind_direction','temperature']
input_variables_lstm = ['wind_avg', 'temperature']
output_variables = ['wind_avg_in24h']

merged_df_scaled = functions_single.normalize_dataset(merged_df, input_variables=input_variables, output_variables=output_variables)
merged_df_scaled_lstm = functions_single.normalize_dataset(merged_df, input_variables=input_variables_lstm, output_variables=output_variables)
merged_df_scaled_mlp = functions_single.normalize_dataset(merged_df, input_variables=input_variables_mlp, output_variables=output_variables)
ratio = 0.6
training_x, training_y, test_x, test_y = functions_single.split_dataset(merged_df_scaled, input_variables=input_variables, output_variables=output_variables, ratio=ratio)
training_x_mlp, training_y_mlp, test_x_mlp, test_y_mlp = functions_single.split_dataset(merged_df_scaled_mlp, input_variables=input_variables_mlp, output_variables=output_variables, ratio=ratio)
training_x_lstm, training_y_lstm, test_x_lstm, test_y_lstm = functions_single.split_dataset(merged_df_scaled_lstm, input_variables=input_variables_lstm, output_variables=output_variables, ratio=ratio)


merged_df_length = len(merged_df)
training_length = int(ratio * merged_df_length)


model_MLP = Sequential()
model_MLP.add(Dense(100, input_shape=(1, len(input_variables_mlp) )))
model_MLP.add(GaussianNoise(0.1))
model_MLP.add(Dense(500 ))
model_MLP.add(Dropout(0.5))
model_MLP.add(Dense(500 ))
model_MLP.add(Dense(100 ))
model_MLP.add(Flatten())
model_MLP.add(Dense(len(output_variables), activation='sigmoid'))
model_MLP.compile(loss='mean_squared_error', optimizer='adam')
model_MLP.fit(training_x_mlp, training_y_mlp, epochs=epochs, batch_size=20, verbose=2)



train_predict_MLP = model_MLP.predict(training_x_mlp)
test_predict_MLP = model_MLP.predict(test_x_mlp, batch_size=20)

train_predict_plot_MLP = train_predict_MLP[:,0]
train_predict_plot_MLP = numpy.reshape(train_predict_plot_MLP, ( training_length, 1))

test_predict_plot_MLP = numpy.empty(merged_df_length)
test_predict_plot_MLP[:] = numpy.nan
test_predict_plot_MLP[len(train_predict_plot_MLP)+1:merged_df_length] = test_predict_MLP[:,0]


model_LSTM = Sequential()
model_LSTM.add(LSTM(50, return_sequences=True, input_shape=(1, len(input_variables_lstm) )))
model_LSTM.add(LSTM(50 ))
model_LSTM.add(Dense(len(output_variables), activation='sigmoid'))
model_LSTM.compile(loss='mean_squared_error', optimizer='adam')
model_LSTM.fit(training_x_lstm, training_y_lstm, epochs=epochs, batch_size=20, verbose=2)

train_predict_LSTM = model_LSTM.predict(training_x_lstm)
test_predict_LSTM = model_LSTM.predict(test_x_lstm, batch_size=20)


train_predict_plot_LSTM = train_predict_LSTM[:,0]
train_predict_plot_LSTM = numpy.reshape(train_predict_plot_LSTM, ( training_length, 1))

test_predict_plot_LSTM = numpy.empty(merged_df_length)
test_predict_plot_LSTM[:] = numpy.nan
test_predict_plot_LSTM[len(train_predict_plot_LSTM)+1:merged_df_length] = test_predict_LSTM[:,0]

train_predict_LSTM_reshaped = train_predict_LSTM.reshape(train_predict_LSTM.shape[0], 1)
train_predict_MLP_reshaped = train_predict_MLP.reshape(train_predict_MLP.shape[0], 1)
train_hybrid = numpy.dstack([train_predict_LSTM_reshaped, train_predict_MLP_reshaped, training_x])
# print(train_hybrid.shape)

test_predict_LSTM_reshaped = test_predict_LSTM.reshape(test_predict_LSTM.shape[0], 1)
test_predict_MLP_reshaped = test_predict_MLP.reshape(test_predict_MLP.shape[0], 1)
test_hybrid = numpy.dstack([test_predict_LSTM_reshaped, test_predict_MLP_reshaped, test_x])



model_HYBRID = Sequential()
model_HYBRID.add(Dense(100, input_shape=( 1, train_hybrid.shape[2] )))
model_HYBRID.add(Dropout(0.5))
model_HYBRID.add(Flatten())
model_HYBRID.add(Dense(1, activation='sigmoid'))
model_HYBRID.summary()
model_HYBRID.compile(loss='mean_squared_error', optimizer='adam')
model_HYBRID.fit(train_hybrid, training_y, epochs=epochs, batch_size=20, verbose=2)


train_predict_HYBRID = model_HYBRID.predict(train_hybrid)
test_predict_HYBRID = model_HYBRID.predict(test_hybrid)

train_predict_plot_HYBRID= train_predict_HYBRID[:,0]
train_predict_plot_HYBRID = numpy.reshape(train_predict_plot_HYBRID, ( training_length, 1))

test_predict_plot_HYBRID = numpy.empty(merged_df_length)
test_predict_plot_HYBRID[:] = numpy.nan
test_predict_plot_HYBRID[len(train_predict_plot_HYBRID)+1:merged_df_length] = test_predict_HYBRID[:,0]

scaled_wind_avg = numpy.array(merged_df_scaled['wind_avg_in24h'])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(training_y[:,0], train_predict_MLP[:,0]))
print('MLP train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_y[:,0], test_predict_MLP[:,0]))
print('MLP test Score: %.2f RMSE' % (testScore))

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(training_y[:,0], train_predict_LSTM[:,0]))
print('LSTM train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_y[:,0], test_predict_LSTM[:,0]))
print('LSTM test Score: %.2f RMSE' % (testScore))

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(training_y[:,0], train_predict_HYBRID[:,0]))
print('Hybrid train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_y[:,0], test_predict_HYBRID[:,0]))
print('Hybrid test Score: %.2f RMSE' % (testScore))

# calculate root mean absolute error
trainScore = mean_absolute_error(training_y[:,0], train_predict_HYBRID[:,0])
print('Hybrid train Score: %.2f MAE' % (trainScore))
testScore = mean_absolute_error(test_y[:,0], test_predict_HYBRID[:,0])
print('Hybrid test Score: %.2f MAE' % (testScore))

plt.plot(scaled_wind_avg, "k")
plt.plot(train_predict_plot_LSTM, "c")
plt.plot(train_predict_plot_MLP, "y")
plt.plot(test_predict_plot_LSTM, "b")
plt.plot(test_predict_plot_MLP, "r--")
plt.plot(test_predict_plot_HYBRID, "g--")

plt.show()




