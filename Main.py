import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from random import randint
from algorithm.GradientDescending import GradientDescending

from algorithm.LeastSquares import LeastSquare

def readData(filename):
    data = pd.read_csv(filename)
    brutData = data

    month = data["Date"].str.split("-").str[1]
    month = pd.to_numeric(month)

    location = data["Location"].astype("category").cat.codes
    location = pd.Series(location)

    median = data["MinTemp"].median()
    data["MinTemp"].fillna(median, inplace=True)
    minTemp = data["MinTemp"]

    median = data["MaxTemp"].median()
    data["MaxTemp"].fillna(median, inplace=True)
    maxTemp = data["MaxTemp"]

    median = data["Rainfall"].median()
    data["Rainfall"].fillna(median, inplace=True)
    rainfall = data["Rainfall"]

    median = data["Evaporation"].median()
    data["Evaporation"].fillna(median, inplace=True)
    evaporation = data["Evaporation"]

    median = data['Sunshine'].median()
    data['Sunshine'].fillna(median, inplace=True)
    sunshine = data['Sunshine']

    windGustDir = data["WindGustDir"].astype("category").cat.codes
    windGustDir = pd.Series(windGustDir)
    #print("WindGustDir: positives / negatives: ", getPercentages(windGustDir))
    #positives / negatives:  (86.58944039598515, 13.41055960401485)
    windGustDir[windGustDir < 0] = avgForPositives(windGustDir)

    median = data["WindGustSpeed"].median()
    data["WindGustSpeed"].fillna(median, inplace=True)
    windGustSpeed = data["WindGustSpeed"]

    windDir9am = data["WindDir9am"].astype("category").cat.codes
    windDir9am = pd.Series(windDir9am)
    #print("WindDir9am: positives / negatives: ", getPercentages(windDir9am))
    #WindDir9am: positives / negatives:  (86.42788395435171, 13.572116045648286)
    windDir9am[windDir9am < 0] = avgForPositives(windDir9am)

    windDir3pm = data["WindDir3pm"].astype("category").cat.codes
    windDir3pm = pd.Series(windDir3pm)
    #print("WindDir3pm: positives / negatives: " , getPercentages(windDir3pm))
    #WindDir3pm: positives / negatives:  (91.26907740959714, 8.730922590402855)
    windDir3pm[windDir3pm < 0] = avgForPositives(windDir3pm)

    median = data["WindSpeed9am"].median()
    data["WindSpeed9am"].fillna(median, inplace=True)
    windSpeed9am = data["WindSpeed9am"]

    median = data["WindSpeed3pm"].median()
    data["WindSpeed3pm"].fillna(median, inplace=True)
    windSpeed3pm = data["WindSpeed3pm"]

    median = data["Humidity9am"].median()
    data["Humidity9am"].fillna(median, inplace=True)
    humidity9am = data["Humidity9am"]

    median = data["Humidity3pm"].median()
    data["Humidity3pm"].fillna(median, inplace=True)
    humidity3pm = data["Humidity3pm"]

    median = data["Pressure9am"].median()
    data["Pressure9am"].fillna(median, inplace=True)
    pressure9am = data["Pressure9am"]

    median = data["Pressure3pm"].median()
    data["Pressure3pm"].fillna(median, inplace=True)
    pressure3pm = data["Pressure3pm"]

    median = data["Cloud9am"].median()
    data["Cloud9am"].fillna(median, inplace=True)
    cloud9am = data["Cloud9am"]

    median = data["Cloud3pm"].median()
    data["Cloud3pm"].fillna(median, inplace=True)
    cloud3pm = data["Cloud3pm"]

    median = data["Temp9am"].median()
    data["Temp9am"].fillna(median, inplace=True)
    temp9am = data["Temp9am"]

    median = data["Temp3pm"].median()
    data["Temp3pm"].fillna(median, inplace=True)
    temp3pm = data["Temp3pm"]

    rainToday = data["RainToday"].astype("category").cat.codes
    rainToday = pd.Series(rainToday)

    rainTomorrow = data["RainTomorrow"].astype("category").cat.codes
    rainTomorrow = pd.Series(rainTomorrow)

    y1 = np.array(rainTomorrow)
    x1 = np.column_stack((month, location, minTemp, maxTemp, rainfall, evaporation, sunshine, 
            windGustDir, windGustSpeed, windDir9am, windDir3pm, windSpeed9am, windSpeed3pm, humidity9am,
            humidity3pm, pressure9am, pressure3pm, cloud9am, cloud3pm, temp9am, temp3pm, rainToday))
    x1 = sm.add_constant(x1, prepend=True)
    predictSample = getPredictSample(x1, brutData)
    x_train, x_test, y_train, y_test = train_test_split(x1, y1)
    return x_train, x_test, y_train, y_test, predictSample

def avgForPositives(data):
    s = 0
    nr = 0
    for el in data:
        if el > 0:
            s = s + el
            nr = nr + 1
    return s * 1.0 / nr

def getPercentages(data):
    positives = 0
    for el in data:
        if el > 0:
            positives = positives + 1
    procent_positives = positives * 100 * 1.0 / len(data)
    procent_negatives = 100.0 - procent_positives
    return procent_positives, procent_negatives

def getPredictSample(preparedData, brutData):
    index = randint(0, 140000)
    print("index: ", index)
    print("brutDataRainTomorrow: ", brutData["RainTomorrow"][index])
    print("##########################################################")
    return preparedData[index]

def leastSquares(x_train, x_test, y_train, y_test, predictSample):
    ls = LeastSquare(x_train, y_train, x_test, y_test)
    #ls.minMaxNormalisation()
    print("Tool: Rain tomorrow?   :", ls.predictTool(predictSample))
    print("Manual: Rain tomorrow? :", ls.predictManual(predictSample))
    ls.accuracyManual()
    ls.accuracyTool()
    print("##########################################################")

def gradientDescending(x_train, x_test, y_train, y_test, predictSample, noIter, learningRate):
    gd = GradientDescending(x_train, y_train, x_test, y_test, noIter, learningRate)
    #print("Tool: Rain tomorrow?  ", gd.predictTool(predictSample))
    print("Manual: Rain tomorrow?", gd.predictManual(predictSample))
    #gd.accuracyManual()
    gd.accuracyTool()
    print("##########################################################")

def main():
    x_train, x_test, y_train, y_test, predictSample = readData("./data/weatherAUS.csv")
    #leastSquares(x_train, x_test, y_train, y_test, predictSample)
    gradientDescending(x_train, x_test, y_train, y_test, predictSample, 100, 0.01)

main()