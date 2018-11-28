# ann-windspeed-forecast
Ensemble ANN with two ensembles a MLP and LSTM network for wind speed prediction from data collected with windGURU stations

To run the predictions run directly one of theese files:

### single-LSTM.py
trains the LSMT network on shifted measurement data and predict on splitted test dataset and draws the plot
### single-MLP.py
trains the MLP network on inputs from WRF forecast and expected outputs of measurements, predicts on splitted test dataset and draws the plot
### single-Ensemble-MLP.py
trains the same architectures as in single-LSTM.py and single-MLP.py on a training set and predicts for a test set. Then trains the ensemble MPL network on a training set and predicts a test set and then draws the plot.


## /data/ folder
The folder contains the measurements and WRF9km forecasts for four experimental spots.
Note: Files only cointain structures of the .csv files, permission to release the datasets have not been gained yet.

The measurements files are pretty straightforward, but the WRF forecast files cointain under the rows with id_var

id_var=1 : wind speed
id_var=2 : wind direction
id_var=3 : temperature

The forecast is placed in the h[n] collun

column h1 : Forecast for hour +1 from the initialization time in 'datum' collumn

column h2 : Forecast for hour +2 from the initialization time in 'datum' collumn

column h3 : Forecast for hour +3 from the initialization time in 'datum' collumn

...

...

column h78 : Forecast for hour +78 from the initialization time in 'datum' collumn
