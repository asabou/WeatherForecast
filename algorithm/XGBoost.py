import pandas
import numpy
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

big_set = pandas.read_csv('weatherAUS.csv') #intreg setul de date
rain = big_set.iloc[:4000,:] # primele 4000 randuri
rain = rain[rain['RainToday'].notna()] 
rain = rain[rain['RainTomorrow'].notna()] # eliminam valorile nule

# rain['Date'] = pandas.to_datetime(rain['Date'])
# rain['year'] = rain['Date'].dt.year
# rain['month'] = rain['Date'].dt.month
# rain['day'] = rain['Date'].dt.day
rain.drop('Date', axis = 1, inplace = True) 

features_with_outliers = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'WindGustSpeed','WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
for feature in features_with_outliers:
    q1 = rain[feature].quantile(0.25)
    q3 = rain[feature].quantile(0.75)
    IQR = q3-q1
    lower_limit = q1 - (IQR*1.5)
    upper_limit = q3 + (IQR*1.5)
    rain.loc[rain[feature]<lower_limit,feature] = lower_limit
    rain.loc[rain[feature]>upper_limit,feature] = upper_limit
#eliminam outliars

rain['RainToday'].replace({'No':0, 'Yes': 1}, inplace = True)
rain['RainTomorrow'].replace({'No':0, 'Yes': 1}, inplace = True)
#inlocuim cu valori intregi

rain.drop('WindGustDir',axis=1,inplace=True)
rain.drop('WindDir9am',axis=1,inplace=True)
rain.drop('WindDir3pm',axis=1,inplace=True)
rain.drop('Location',axis=1,inplace=True)
#eliminam coloanele (*)

rain['RainToday'] = rain['RainToday'].astype(int) 
rain['RainTomorrow'] = rain['RainTomorrow'].astype(int)

X = rain.drop(['RainTomorrow'],axis=1)
y = rain['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
#impartim seturile de date in train si test

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
#folosim XGBClassifier pentru predictii

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#masuram acuratetea (~85%)

print("Length of Training Data: {}".format(len(X_train)))
print("Length of Testing Data: {}".format(len(X_test)))

