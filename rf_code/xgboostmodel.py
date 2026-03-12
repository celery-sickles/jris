import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


#Load data
df = pd.read_csv("C:/Users/alyss/Downloads/OH260225190059543S845PK/CrashStatistics.csv")
df.drop(columns=['LocalReportNumber','DocumentNumber','HitSkip', 'SecondaryCrash','UnitInError','County','FIPSPlaceCode', 'PhotosTaken', 'OH2', 'OH3','OH1P','OHOther','PrivateProperty','ReportingAgencyNCIC','Narrative','ReportTakenBy','Supplement','CrashReportedDateTime','DispatchedDateTime','ArrivedDateTime','SceneClearedDateTime','OtherInvestigationTime','OfficerName','OfficerBadgeNumber','CheckedByOfficerName','CheckedByBadgeNumber'], inplace=True)
#print(list(df))
#X values: CrashDateTime, Weather, LightCondition
#Other possible X values: LocationFirstHarmfulEvent, IntersectionOrApproachRelated/Number of Approaches, WithinInterchangeArea, Manner of Collision
#Y values: CrashSeverity, SecondaryCrash, NumberOfUnits
#Clean distance from reference data

#Things I am thinking of: 
# Find some way to normalize location data - or perhaps don't include location data?
# Instead of creating an "accident - no accident" model, create a severity model?
# If I wanted to create an accident classification model I would also have to have traffic data for all the traffic that occurred and I don't have that.

#Convert and split "Time" to int values
df["CrashDateTime"] = pd.to_datetime(df["CrashDateTime"])
df["Year"] = df["CrashDateTime"].dt.year
df["Month"] = df["CrashDateTime"].dt.month
df["Day"] = df["CrashDateTime"].dt.day
df["Hour"] = df["CrashDateTime"].dt.hour
df["Minute"] = df["CrashDateTime"].dt.minute
df = df.drop(columns=["CrashDateTime"])
#print(list(df))

#Encode "Location" to a categorical value
le = LabelEncoder()
#df['Location'] = le.fit_transform(df['Location'])
df['CrashSeverity'] = le.fit_transform(df['CrashSeverity'])
df['LightCondition'] = le.fit_transform(df['LightCondition'])
df['Weather'] = le.fit_transform(df['Weather'])
df['MannerOfCollision'] = le.fit_transform(df['MannerOfCollision'])
df['RoadwayDivided'] = le.fit_transform(df['RoadwayDivided'])



#Create lists of feature and target columns (x feature, y target)
x = df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'NumberOfUnits','LightCondition','Weather','MannerOfCollision', 'RoadwayDivided']]
y = df['CrashSeverity']


#Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


# == Using XGBoost to predict accidents ==

#Set base parameters for XGBoost model
model_1 = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# == Feature selection ==

#sbs = SFS(model, k_features=10, forward=False, floating=False, scoring='accuracy')
# Fitting the SBS model to the training data (X_train and y_train)
#sbs = sbs.fit(X_train, y_train)
#selected_features = X_train.columns[sbs.get_support()]

#print(selected_features)


model_1.fit(X_train, y_train)
y_pred = model_1.predict(X_test)


#Built in feature importance
model_2 = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

importance = model_2.feature_importances_

feature_importance = pd.DataFrame({
    "feature": x.columns,
    "importance": importance
}).sort_values(by="importance", ascending=False)

print(feature_importance)

top_features = feature_importance.head(10)["feature"]

model_2.fit(X_train[top_features], y_train)
y_pred = model_2.predict(X_test)

#print("XGBoost prediction: ")
#print(*y_pred)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Squared Error: " + str(mse))
print("Root Mean Squared Error: " + str(rmse))


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
#print("Predicted accidents for Intersection A at 17:30 on 2023-06-01 with 100 vehicles: ",
#      predict_accidents("Intersection A", "2023-06-01 17:30", 100))
