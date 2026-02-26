import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np

#Load data
df = pd.read_csv("C:/Users/alyss/Downloads/traffic_data.csv")
#print(df)

#Convert and split "Time" to numerical values
df["Time"] = pd.to_datetime(df["Time"])
df["Year"] = df["Time"].dt.year
df["Month"] = df["Time"].dt.month
df["Day"] = df["Time"].dt.day
df["Hour"] = df["Time"].dt.hour
df["Minute"] = df["Time"].dt.minute
df = df.drop(columns=["Time"])
#print(df)

#Encode "Location" to a categorical value
le = LabelEncoder()
df['Location'] = le.fit_transform(df['Location'])

#Create lists of feature and target columns (x feature, y target)
x = df[['Location', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Vehicles']]
y = df['Accidents']

#Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

#Next: build XGBoost model, build function to predict accidents based on specified input data