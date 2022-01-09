import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

class LeastSquare:
    def __init__(self, trainInputs, trainOutputs, testInputs, testOutputs):
        self.__trainInputs = trainInputs
        self.__trainOutputs = trainOutputs 
        self.__testInputs = testInputs 
        self.__testOutputs = testOutputs

    def __coeficients(self):
        X = self.__trainInputs
        Y = self.__trainOutputs
        XT = np.transpose(X)
        XTxX = np.dot(XT,X)
        inv = np.linalg.inv(XTxX)
        B = np.dot(inv,XT)
        B = np.dot(B,Y)
        return B

    def manualErrorPrediction(self):
        B = self.__coeficients()
        X = self.__testInputs
        Y = self.__testOutputs
        F = np.dot(X,B)
        Diff = Y - F
        sume = sum([x*x for x in Diff])
        return sume / len(X)

    def toolErrorPrediction(self):
        regressor = linear_model.LinearRegression()
        regressor.fit(self.__trainInputs, self.__trainOutputs)
        F = regressor.predict(self.__testInputs)
        return mean_squared_error(self.__testOutputs,F)

    def elimConstant(self, inputs):
        inputs_new = []
        for line in inputs:
            line_new = line[1:]
            inputs_new.append(line_new)
        return np.array(inputs_new)
    
    def accuracyTool(self):
        print("tool  : ", 100.0 - (self.toolErrorPrediction() * 100))
    
    def accuracyManual(self):
        print("manual: ", 100.0 - (self.manualErrorPrediction() * 100))

    def predictTool(self, data):
        regressor = linear_model.LinearRegression()
        regressor.fit(self.__trainInputs, self.__trainOutputs)
        F = regressor.predict([data])
        print("F tool:  ", F[0])
        if F[0] < 0.5:
            return "No"
        return "Yes"

    def predictManual(self, data):
        regressor = np.dot(data, self.__coeficients())
        print("F manual: ", regressor)
        if regressor < 0.5:
            return "No"
        return "Yes"
    
    def minMaxNormalisation(self):
        x_train = self.elimConstant(self.__trainInputs)
        x_test = self.elimConstant(self.__testInputs)
        y_train = self.__trainOutputs
        y_test = self.__testOutputs
        y_train = np.reshape(y_train, (-1, 1)) # datele din y se pun pe verticala
        y_test = np.reshape(y_test, (-1, 1))
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaler_x.fit(x_train)
        xtrain_scale = scaler_x.transform(x_train)
        scaler_x.fit(x_test)
        xtest_scale = scaler_x.transform(x_test)
        scaler_y.fit(y_train)
        ytrain_scale = scaler_y.transform(y_train)
        scaler_y.fit(y_test)
        ytest_scale = scaler_y.transform(y_test)
        self.__trainInputs = sm.add_constant(xtrain_scale, prepend=True)
        self.__testInputs = sm.add_constant(xtest_scale, prepend=True)
        self.__trainOutputs = ytrain_scale
        self.__testOutputs = ytest_scale
