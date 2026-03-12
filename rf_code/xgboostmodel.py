import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
import numpy as np



#ML pipeline steps:

#Preprocessing: ignore null values, encode categorical variables, standardize variables (onehotencoder)
#Feature selection
#Cross validation
#Hyperparameter optimization

#Load data
df = pd.read_csv("C:/Users/alyss/Downloads/OH260225190059543S845PK/CrashStatistics.csv")
df.drop(columns=['LocalReportNumber','DocumentNumber','HitSkip', 'SecondaryCrash','UnitInError','County','FIPSPlaceCode', 'PhotosTaken', 'OH2', 'OH3','OH1P','OHOther','PrivateProperty','ReportingAgencyNCIC','Narrative','ReportTakenBy','Supplement','CrashReportedDateTime','DispatchedDateTime','ArrivedDateTime','SceneClearedDateTime','OtherInvestigationTime','OfficerName','OfficerBadgeNumber','CheckedByOfficerName','CheckedByBadgeNumber'], inplace=True)
#print(list(df))
#X values: CrashDateTime, Weather, LightCondition
#Other possible X values: LocationFirstHarmfulEvent, IntersectionOrApproachRelated/Number of Approaches, WithinInterchangeArea, Manner of Collision
#Y values: CrashSeverity, SecondaryCrash, NumberOfUnits
#Clean distance from reference data

#Convert and split "Time" to int values
df["CrashDateTime"] = pd.to_datetime(df["CrashDateTime"])
df["Year"] = df["CrashDateTime"].dt.year
df["Month"] = df["CrashDateTime"].dt.month
df["Day"] = df["CrashDateTime"].dt.day
df["Hour"] = df["CrashDateTime"].dt.hour
df["Minute"] = df["CrashDateTime"].dt.minute
df = df.drop(columns=["CrashDateTime"])
#print(list(df))

#Label encoding on y
le = LabelEncoder()
df['CrashSeverity'] = le.fit_transform(df['CrashSeverity'])


#Create lists of feature and target columns (x feature, y target)
x = df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'NumberOfUnits','LightCondition','Weather','MannerOfCollision', 'RoadwayDivided', "IntersectionOrApproachRelated","NumberOfApproaches","WithinInterchangeArea","Latitude","Longitude"]]
y = df['CrashSeverity']


#Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

#Preprocessing: Automatically convert and encode categorical columns

#Fill all unknown values with 0

#Select numerical and categorical columns
numeric_features = x.select_dtypes(include="number").columns
categorical_features = x.select_dtypes(exclude="number").columns
#print(type(numeric_features))
#print(categorical_features)

#Transform data
from sklearn.pipeline import Pipeline
import sklearn.preprocessing as pre
from sklearn.compose import ColumnTransformer

#("imputer", SimpleImputer(strategy="median")),
data_transformer = ColumnTransformer(
  transformers = [
    ('rescale numeric', pre.StandardScaler(), numeric_features),
    ('recode categorical', 
      pre.OneHotEncoder(handle_unknown = 'ignore'), 
      categorical_features)
    ])
    
#Set base parameters for XGBoost model (to use in pipeline) (adjust parameters with hyperoptimization later)
model_1 = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

#Feature selection
from sklearn.feature_selection import SelectFromModel
feature_selector = SelectFromModel(
    estimator=XGBClassifier(
        n_estimators=200,
        eval_metric="logloss",
        random_state=42
    ),
    threshold="median"
)

print(feature_selector.prefit(X_train, y_train))

from sklearn.pipeline import Pipeline
xgb_pipeline =  Pipeline(steps = [
    ('preprocessing', data_transformer),
    ("feature_selection", feature_selector),
    ('XGB', model_1)
    ])

#Running the model: 
xgb_pipeline.fit(X_train, y_train)
prediction = xgb_pipeline.predict(X_test)
print(*prediction)
print("Test accuracy:", xgb_pipeline.score(X_test, y_test))
#y_pred = model_1.predict(X_test)

#print("XGBoost prediction: ")
#print(*y_pred)



#mse = mean_squared_error(y_test, y_pred)
#rmse = np.sqrt(mse)

#print("Mean Squared Error: " + str(mse))
#print("Root Mean Squared Error: " + str(rmse))

