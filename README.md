# ann-windspeed-forecast
Ensemble ANN with two ensembles a MLP and LSTM network for wind speed prediction

To run the predictions run directly one of theese files:

### single-LSTM.py
trains the LSMT network on shifted measurement data and predict on splitted test dataset and draws the plot
### single-MLP.py
trains the MLP network on inputs from WRF forecast and expected outputs of measurements, predicts on splitted test dataset and draws the plot
### single-Ensemble-MLP.py
trains the same architectures as in single-LSTM.py and single-MLP.py on a training set and predicts for a test set. Then trains the ensemble MPL network on a training set and predicts a test set and then draws the plot.
