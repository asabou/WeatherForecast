import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

class LeastSquare:
    def __init__(self, trainInputs, trainOutputs, testInputs, testOutputs):
        self.__trainInputs = trainInputs
        self.__trainOutputs = trainOutputs 
        self.__testInputs = testInputs 
        self.__testOutputs = testOutputs

    def __coeficients(self):
        trainInputs = []
        for x in self.__trainInputs:
            line = [1]
            for xx in x:
                line.append(xx)
            trainInputs.append(line)
        X = trainInputs
        Y = self.__trainOutputs
        XT = np.transpose(X)
        XTxX = np.dot(XT,X)
        inv = np.linalg.inv(XTxX)
        B = np.dot(inv,XT)
        B = np.dot(B,Y)
        return B

    def manualErrorPrediction(self):
        B = self.__coeficients()
        testInputs = []
        for x in self.__testInputs:
            line = [1]
            for xx in x:
                line.append(xx)
            testInputs.append(line)
        X = testInputs
        Y = self.__testOutputs
        F = np.dot(X,B)
        Diff = Y - F
        sume = sum([x*x for x in Diff])
        return sume / len(X)

    def toolErrorPrediction(self):
        regressor = linear_model.LinearRegression()
        regressor.fit(self.__trainInputs,self.__trainOutputs)
        F = regressor.predict(self.__testInputs)
        return mean_squared_error(self.__testOutputs,F)
