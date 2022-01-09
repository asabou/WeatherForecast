from math import exp
import numpy as np
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def read_data():
    train_df = pd.read_csv('weatherAUS.csv')
    a = train_df.head()
    b = train_df
    # in fisierul din care citim datele, avem diferite tipuri de date(float,int,boolean, string)
    # in reteaua neuronala ne folosim doar de tipurile de tip float, int
    # astfel ca pe celelalte trebuie sa le mapam, transformandu stringurile si valorile booleene in valori intregi
    # mai avem in setul de date valori NaN care semnaleaza faptul ca lipseste acea valoare
    # pentru acest caz folosim media aritmetica a valorilor de pe coloana respectiva si inlocuim
    # valorile NaN cu acea valoare

    # y-ul contine raintoday-ul
    rain = train_df['RainTomorrow'].astype("category").cat.codes
    rain = pd.Series(rain)

    # pentru x ar trebui sa iau si luna, din coloana dat3e
    # datele de care avem nevoie pentru a contrui x-ul
    month = train_df['Date'].str.split("-").str[1]
    month = pd.to_numeric(month)

    # pentru datele care contin stringuri, trebuie sa le atribuim o valoare de tip int
    # folosim astype("category") pentru a face acest lucru
    location = train_df['Location'].astype("category").cat.codes
    location = pd.Series(location)

    # aici ne ocumam de completarea valorilor lipsa
    # cu ajutorul functiilor din panda
    median = train_df['MinTemp'].median()
    train_df['MinTemp'].fillna(median, inplace=True)
    minTemp = train_df['MinTemp']

    median = train_df['MaxTemp'].median()
    train_df['MaxTemp'].fillna(median, inplace=True)
    maxTemp = train_df['MaxTemp']

    median = train_df['Rainfall'].median()
    train_df['Rainfall'].fillna(median, inplace=True)
    rainfall = train_df['Rainfall']

    median - train_df['Sunshine'].median()
    train_df['Sunshine'].fillna(median, inplace=True)
    sunshine = train_df['Sunshine']  # 48% valori Na

    median = train_df['WindGustSpeed'].median()
    train_df['WindGustSpeed'].fillna(median, inplace=True)
    windGustSpeed = train_df['WindGustSpeed']

    median = train_df['WindSpeed9am'].median()
    train_df['WindSpeed9am'].fillna(median, inplace=True)
    windSpeed9am = train_df['WindSpeed9am']

    median = train_df['WindSpeed3pm'].median()
    train_df['WindSpeed3pm'].fillna(median, inplace=True)
    windSpeed3pm = train_df['WindSpeed3pm']

    median = train_df['Humidity9am'].median()
    train_df['Humidity9am'].fillna(median, inplace=True)
    humidity9am = train_df['Humidity9am']

    median = train_df['Humidity3pm'].median()
    train_df['Humidity3pm'].fillna(median, inplace=True)
    humidity3pm = train_df['Humidity3pm']

    median = train_df['Pressure9am'].median()
    train_df['Pressure9am'].fillna(median, inplace=True)
    pressure9am = train_df['Pressure9am']

    median = train_df['Pressure3pm'].median()
    train_df['Pressure3pm'].fillna(median, inplace=True)
    pressure3pm = train_df['Pressure3pm']

    median = train_df['Cloud9am'].median()
    train_df['Cloud9am'].fillna(median, inplace=True)
    cloud9am = train_df['Cloud9am']

    median = train_df['Cloud3pm'].median()
    train_df['Cloud3pm'].fillna(median, inplace=True)
    cloud3pm = train_df['Cloud3pm']

    median = train_df['Temp9am'].median()
    train_df['Temp9am'].fillna(median, inplace=True)
    temp9am = train_df['Temp9am']

    median = train_df['Temp3pm'].median()
    train_df['Temp3pm'].fillna(median, inplace=True)
    temp3pm = train_df['Temp3pm']

    y1 = np.array(rain)
    # contruim x-ul
    x1 = np.column_stack((location, minTemp, maxTemp, rainfall, sunshine, windGustSpeed, windSpeed9am, windSpeed3pm, humidity9am,
                          humidity3pm, pressure9am, pressure3pm, cloud9am, cloud3pm, temp9am, temp3pm))
    x1 = sm.add_constant(x1, prepend=True)  # se adauga valoarea 1 pe prima coloana din matricea cu date
    x_train, x_test, y_train, y_test = train_test_split(x1, y1)  # 80% foloseste pentru train si 20% pentru testare
    return x_train, x_test, y_train, y_test


def normalizare_min_max(x_train, x_test, y_train, y_test):
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
    return xtrain_scale, xtest_scale, ytrain_scale, ytest_scale

# procesul de normalizare a datelor se face pe fiecare coloana pentru a incadra valorile intr-un anumit interval
def normalizare_date(x_train):
    medii = []
    deviatii = []
    nrLines = len(x_train)
    for j in range(1, len(x_train[0])):
        sum = 0.0
        # calculul mediei
        for i in range(nrLines):
            sum += x_train[i][j]
        medii.append(sum / nrLines) #se calculeaza media pentru fiecare coloana
        sumd = 0.0
        for i in range(nrLines):
            sumd += (x_train[i][j] - medii[-1]) ** 2
        sumd /= (nrLines - 1)
        sumd = sqrt(sumd)
        deviatii.append(sumd)# si deviatia

    for i in range(nrLines):
        for j in range(1, len(x_train[0])):
            x_train[i][j] = float((x_train[i][j] - medii[j - 1]) / deviatii[j - 1]) # se modifica valorile
    return medii, deviatii # se returneaza valorile pentru a normaliza datele de test folosind media si deviatia de la datele de train


def cal_cost(theta, x_train, y_train):
    predictions = x_train.dot(theta)
    cost = np.sum(np.square(predictions - y_train))
    return cost


# pentru metoda asta avem nevoie de un learning rate(lr), pe care il setam de la inceput cu o anumita valoare
# avem nevoie si de un numar de iteratii, pentru a face procesul de determinare a valorilor din beta de nr_iter ori
#
def metoda_gradient_descendent(x_train, y_train, lr, nr_iter):
    #y_train = np.matrix(y_train).T
    n, m = x_train.shape
    beta = np.zeros((m, 1)) # initial luam valorile de beta ca fiind toate 0
    matrix = np.matrix(x_train)
    cost_history = np.zeros(nr_iter) # folosit pentru plot
    for it in range(nr_iter):
        prediction = np.dot(matrix, beta) # face m o predictie, inmultind matricea cu beta-urile actuale
        err = prediction - y_train # determinam eroarea intre predictia facuta anterior si valorile ce trebuiau sa fie defapt
        beta = beta - lr * (np.dot(matrix.T, err)) # recalculam beta-ul cu ajutorul lr-ului si a valorile din err.
        cost_history[it] = cal_cost(beta, x_train, y_train) # folosit pentru plot
    return beta, cost_history


# mai intai se normalizeaza datele
# dupa care se determina y-ul final, inmultind beta-urile determinate de noi cu valorile de x
def calculare_y_final(x_train, x_test, y_test, beta):
    nrLines = len(x_test)
    nrColomn = len(x_test[0])
    #normalizam datele de test
    # for i in range(nrLines):
    #     for j in range(1, nrColomn):
    #         x_test[i][j] = (x_test[i][j] - medii[j - 1]) / deviatii[j - 1]
    X_test = np.matrix(x_test)
    Y = np.dot(X_test, beta)
    return y_test, Y


def meanSquaredError(y_calculat_de_noi, y_test):
    suma = 0
    dif = y_test - y_calculat_de_noi
    for x in range(0, len(dif)):
        suma += ((dif[x][0]))**2
    return suma / len(y_calculat_de_noi)


def main():
    x_train, x_test, y_train, y_test = read_data() # citim datele
    #medii, deviatii = normalizare_date(x_train) # le normalizam ca sa le incadram intr-un anumit interval
    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train))
    x_test = pd.DataFrame(scaler.transform(x_test))
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    lr = 0.000000001 #decidem rata de invatare
    nr_iter = 10000 # decidem numarul de iteratii
    print("AFISARE mean_squaredError pentru metoda gradient descendent")

    x_train = np.matrix(x_train)
    x_test = np.matrix(x_test)
    y_train = np.matrix(y_train)
    y_test = np.matrix(y_test)
    #xtrain_scale, xtest_scale, ytrain_scale, ytest_scale = normalizare_min_max(x_train, x_test, y_train, y_test)
    beta, cost_history = metoda_gradient_descendent(x_train, y_train, lr, nr_iter) # aplicam metoda
    y_test, y_predict = calculare_y_final(x_train, x_test, y_test, beta) # determinam y-ul final
    #y_test, y_predict = calculare_y_final(xtrain_scale, xtest_scale, ytest_scale, beta)  # determinam y-ul final

    print(meanSquaredError(y_predict, y_test)) # verificam cat de buna ii ecuatia noastra y.

    # plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_ylabel('J(Theta)')
    ax.set_xlabel('Iterations')
    _ = ax.plot(range(nr_iter), cost_history, 'b.')
    plt.show()

main()