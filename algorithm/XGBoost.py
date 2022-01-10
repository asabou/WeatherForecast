import pandas
import numpy
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
class XGBoost:
    def __init__(self, x_train, y_train, x_test, y_test):
       self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test
    

    def toolErrorPrediction(self):
        model = XGBClassifier()
        model.fit(self.x_train, self.y_train)
        F = model.predict(self.x_test)
        # predictions = [round(value) for value in F]

        print("F: ", F)
        return mean_squared_error(self.y_test,F)

    def accuracyTool(self):
        print("tool  : ", 100.0 - (self.toolErrorPrediction() * 100))

    def predictTool(self, data):
        model = XGBClassifier()
        model.fit(self.x_train, self.y_train)
        F = model.predict(self.x_test)
        print("F tool:  ", F[0])
        if F[0] < 0.5:
            return "No"
        return "Yes"




