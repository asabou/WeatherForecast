from math import sqrt, pow
import numpy as np

class GradientDescending:
    def __init__(self, params):
        self.__params = params
    
    def __setMatrix(self, matrix):
        mat = []
        for x in matrix:
            line = [1]
            for xx in x:
                line.append(xx)
            mat.append(line)
        return mat

    #fit
    def __coeficients(self):
        X = self.__setMatrix(self.__params["trainInputs"])
        Y = self.__params["trainOutputs"]
        betha = [0.0 for _ in range(len(X[0]))]
        noIter = self.__params["noIter"]
        lr = self.__params["learningRate"]
        for _ in range(noIter):
            prediction = np.dot(X, betha)
            err = prediction - Y
            betha = betha - lr * np.dot(np.transpose(X), err)
        return betha
    
    '''
    def __normalisation(self, matrix):
        averages = []
        deviations = []
        n = len(matrix)
        m = len(matrix[0])
        for j in range(1,m):
            sume = sum([matrix[i][j] for i in range(n)])
            avg = sume / n
            averages.append(avg)
            devSume = sum([pow(matrix[i][j] - avg, 2) for i in range(n)])
            dev = sqrt(devSume / (n-1))
            deviations.append(dev)
        return averages, deviations
    '''

    def errorPrediction(self):
        #avg, dev = self.__normalisation(self.__setMatrix(self.__params["trainInputs"]))
        X = self.__setMatrix(self.__params["testInputs"])
        n = len(X)
        m = len(X[0])
        #for i in range(n):
        #    for j in range(1,m):
        #        X[i][j] = (X[i][j] - avg[j-1])/dev[j-1]
        betha = self.__coeficients()
        F = np.dot(X, betha)
        Y = self.__params["testOutputs"]
        Diff = F - Y
        sume = sum([x*x for x in Diff])
        err = sume / len(Diff)
        return err