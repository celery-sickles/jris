#Random Forest model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('C:/Users/alyss/Downloads/traffic_data.csv')
print(df)

x = df.drop('Location', axis=1)
y = df['Accidents']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



#To install package: first select virtual environment, then close terminal and re-open it

#Convert time to a float
#Encode location
#Figure out how it's selecting multiple features (location, time, and vehicles are the features)