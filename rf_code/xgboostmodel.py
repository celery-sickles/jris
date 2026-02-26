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
print(df)

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


# == Using XGBoost to predict accidents ==

#Set base parameters for XGBoost model
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)


print(model.predict(X_test))


#Create function to run XGBoost for given parameters
def predict_accidents(location, time_str, num_vehicles):
    time_obj = pd.to_datetime(time_str, format='%Y-%m-%d %H:%M')
    year = time_obj.year
    month = time_obj.month
    day = time_obj.day
    hour = time_obj.hour
    minute = time_obj.minute
   
    location_encoded = le.transform([location])[0]
   
    input_data = np.array([[
        location_encoded,
        year,
        month,
        day,
        hour,
        minute,
        num_vehicles
    ]])
    prediction = model.predict(input_data)
   
    return round(prediction[0], 2)


#Sample prediction
print("Predicted accidents for Intersection A at 17:30 on 2023-06-01 with 100 vehicles: ",
      predict_accidents("Intersection A", "2023-06-01 17:30", 100))
