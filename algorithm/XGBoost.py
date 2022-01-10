import pandas
import numpy
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
class XGBoost:
    def __init__(self, trainInputs, trainOutputs, testInputs, testOutputs):
        self.__trainInputs = trainInputs
        self.__trainOutputs = trainOutputs 
        self.__testInputs = testInputs 
        self.__testOutputs = testOutputs
    

    def toolErrorPrediction(self):
        model = XGBClassifier()
        model.fit(self.__trainInputs, self.__trainOutputs)
        F = model.predict(self.__testInputs)
        print("F: ", F)
        return mean_squared_error(self.__testOutputs,F)

    def accuracyTool(self):
        print("tool  : ", 100.0 - (self.toolErrorPrediction() * 100))

    def predictTool(self, data):
        model = XGBClassifier()
        model.fit(self.__trainInputs, self.__trainOutputs)
        F = model.predict(data)
        print("F tool:  ", F[0])
        if F[0] < 0.5:
            return "No"
        return "Yes"



# model = XGBClassifier()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# #folosim XGBClassifier pentru predictii

# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# #masuram acuratetea (~85%)

# print("Length of Training Data: {}".format(len(X_train)))
# print("Length of Testing Data: {}".format(len(X_test)))

