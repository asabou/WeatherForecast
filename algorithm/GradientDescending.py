import numpy as np
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import mean_squared_error
import math

class GradientDescending:
    def __init__(self, trainInputs, trainOutputs, testInputs, testOutputs, noIter, learningRate):
        self.__trainInputs = trainInputs
        self.__trainOutputs = trainOutputs
        self.__testInputs = testInputs
        self.__testOutputs = testOutputs
        self.__noIter = noIter
        self.__learningRate = learningRate

    def __coeficients(self):
        n, m = self.__trainInputs.shape
        B = np.zeros((m,1))
        matrix = np.matrix(self.__trainInputs)
        for _ in range(self.__noIter):
           prediction = np.dot(matrix, B)
           err = prediction - self.__trainOutputs
           B = B - self.__learningRate * (np.dot(matrix.T, err))
        return B
        
    def manualErrorPrediction(self): 
        B = self.__coeficients()
        X = self.__testInputs
        Y = self.__testOutputs
        F = np.dot(X,B)
        Diff = Y - F
        sume = sum([x*x for x in Diff])
        return sume / len(X)

    def accuracyManual(self):
        print("manual: ", 100.0 - (self.manualErrorPrediction() * 100))

    def toolErrorPrediction(self):
        lr = LogisticRegression(max_iter=self.__noIter, penalty='l2')
        lr.fit(self.__trainInputs, self.__trainOutputs)
        F = lr.predict(self.__testInputs)
        return mean_squared_error(self.__testOutputs, F)

    def accuracyTool(self):
        print("tool  : ", 100.0 - (self.toolErrorPrediction() * 100))

    def predictTool(self, data):
        lr = LogisticRegression(max_iter=self.__noIter, penalty='l2')
        lr.fit(self.__trainInputs, self.__trainOutputs)
        F = lr.predict([data])
        print("F tool:  ", F[0])
        if F[0] < 0.5:
            return "No"
        return "Yes"
    
    def predictManual(self, data):
        F = np.dot(data, self.__coeficients())
        print("F manual: ", F)
        if F < 0.5:
            return "No"
        return "Yes"