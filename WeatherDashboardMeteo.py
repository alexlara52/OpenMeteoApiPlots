import pandas as pd
from pandas import json_normalize
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
# This program is based on OpenMeteo's weather api
# get the user longitude and latitude input
userlong = str(input('enter a longitude: '))
userlat = str(input('enter a latitude: '))
# configure based on meteo website data and skip rows so it goes straight into the columns
# each letter represents different data gathered q is observed from era5, forecast is meteo's preferred forecast, and e is ecmwf.
q = pd.read_csv('https://archive-api.open-meteo.com/v1/archive?latitude='+ userlat + '&longitude='+ userlong + '&start_date=2022-01-01&end_date=2023-01-07&hourly=temperature_2m&format=csv', skiprows=[0,1,2])
t = pd.read_csv('https://api.open-meteo.com/v1/forecast?latitude='+ userlat + '&longitude='+ userlong + '&past_days=30&hourly=temperature_2m&format=csv', skiprows=[0,1,2])
e = pd.read_csv('https://api.open-meteo.com/v1/ecmwf?latitude='+ userlat + '&longitude='+ userlong + '&past_days=30&hourly=temperature_2m&format=csv', skiprows=[0,1,2])
# rename columns ex temperature
q.rename(columns={q.columns[1]: 'temperatureobs'},inplace=True)
t.rename(columns={t.columns[1]: 'temperaturemet'},inplace=True)
e.rename(columns={e.columns[1]: 'temperatureecm'},inplace=True)
# get the absolute value since they were in negatives for some odd reason, need to check why data is not correct.
q["temperatureobs"] = abs(q["temperatureobs"])
t["temperaturemet"] = abs(t["temperaturemet"])
e["temperatureecm"] = abs(e["temperatureecm"])
# put to datetime
q["time"] = pd.to_datetime(q["time"])
t["time"] = pd.to_datetime(t["time"])
e["time"] = pd.to_datetime(e["time"])
# check dataframe to make sure data matches up with website (currently does not)
print(e)
# merge so all the data is in one dataframe
d = pd.merge_asof(e, t, on='time', direction='nearest')
r = pd.merge_asof(q, d, on='time', direction='nearest')
# configure and plot data
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
r.plot(x="time", y=["temperatureobs", "temperaturemet","temperatureecm"],
        kind="line", figsize=(10, 10))
# show plot
plt.show()
# create a new dataframe
dfq = q
# change time from datetime so that it can run through the ML model
dfq['time'] = pd.to_numeric(dfq['time'].astype(np.int64))
# Create the features and target arrays
X = dfq[['time']]
y = dfq['temperatureobs']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate a Random Forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf.fit(X_train, y_train)

# Use the model to predict temperature values
y_pred = rf.predict(X_test)

# Score the model
score = rf.score(X_test, y_test)
print("Model score:", score)
# plot observed vs predicted
df_plot = pd.DataFrame({'time':X_test['time'],'observed':y_test,'predicted':y_pred})
# get time back to datetime for plot
df_plot['time'] = pd.to_datetime(df_plot['time'])
# plot parameters
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
df_plot.plot(x="time", y=["observed", "predicted"],
        kind="line", figsize=(10, 10))
# show plot
plt.show()