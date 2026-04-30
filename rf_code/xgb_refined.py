

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

cols_to_use = ['CrashSeverity', 'CrashDateTime', 'NumberOfUnits', 'LightCondition', 'Weather', 'MannerOfCollision', 'RoadwayDivided', 'IntersectionOrApproachRelated', 'NumberOfApproaches', 'WithinInterchangeArea', 'Latitude', 'Longitude','RoadContour','RoadCondition','RoadSurface', 'AnimalRelated','AnimalDeerRelated','AlcoholRelated','DrugRelated','BicycleRelated','MotorCycleRelated','SpeedRelated','PedestrianRelated',	'SemiTruckRelated','SmallTruckRelated','YouthRelated','TeenRelated','DUI21Related','SeniorRelated',	'CommercialRelated','CommercialAtFault']

df = pd.read_csv("C:/Users/alyss/Downloads/OH260225190059543S845PK/CrashStatistics.csv", usecols=cols_to_use, low_memory=False)


df["CrashDateTime"] = pd.to_datetime(df["CrashDateTime"])
df["Year"] = df["CrashDateTime"].dt.year
df["Month"] = df["CrashDateTime"].dt.month
df["Day"] = df["CrashDateTime"].dt.day
df["Hour"] = df["CrashDateTime"].dt.hour
df["Minute"] = df["CrashDateTime"].dt.minute
df = df.drop(columns=["CrashDateTime"])

le = LabelEncoder()

SeverityOrder = {'Property Damage Only': 0, 'Injury Possible': 1, 'Minor Injury Suspected': 2, 'Serious Injury Suspected': 3, 'Fatal': 4}
df['EncodedCrashSeverity'] = df['CrashSeverity'].map(SeverityOrder)

label_map = dict(zip(df['CrashSeverity'], df['EncodedCrashSeverity']))


x = df.drop(columns=['CrashSeverity', 'EncodedCrashSeverity'])
y = df['EncodedCrashSeverity']


#Split testing and training data
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)


#Separate numeric and categorical features
numeric_features = x.select_dtypes(include="number").columns
categorical_features = x.select_dtypes(exclude="number").columns
print(numeric_features)
print(categorical_features)



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
#    ('imputer', SimpleImputer(strategy='most_frequent')), # error in manner_of_collision for angle
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

from sklearn.compose import ColumnTransformer

data_transformer = ColumnTransformer(
  transformers = [
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, 
      categorical_features)
    ])

transformed_data = data_transformer.fit_transform(X_train)

#column_names = data_transformer.get_feature_names_out()

#Feature selection
from sklearn.feature_selection import SelectFromModel
feature_selector = SelectFromModel(
    estimator=XGBClassifier(
        n_estimators=50,
        eval_metric="logloss", #can remove eval_metric?
        random_state=42
    ),
    threshold="median"
)


model_1 = XGBClassifier(
    n_estimators=75,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb_pipeline =  Pipeline(steps = [
    ('preprocessing', data_transformer),
    ("feature_selection", feature_selector),
    ('XGB', model_1)
    ])


#Make weights count to account for uneven split

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

#sklearn.set_config(enable_metadata_routing=True)

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights_dict = dict(zip(np.unique(y_train), weights))
sample_weights = np.array([class_weights_dict[cls] for cls in y_train])

xgb_pipeline = xgb_pipeline.fit(X_train, y_train, XGB__sample_weight=sample_weights)
y_pred = xgb_pipeline.predict(X_test)

