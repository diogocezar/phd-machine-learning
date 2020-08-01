import sys
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
import pylab as pl
import seaborn as sns
import time
import json


def run(data, normalized, distance, k):
    print("Loading data...")
    X_data, y_data = load_svmlight_file(data + ".txt")
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.5, random_state=5)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    if normalized:
        scaler = preprocessing.MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    neigh = KNeighborsClassifier(n_neighbors=k, metric=distance)

    neigh.fit(X_train, y_train)

    y_pred = neigh.predict(X_test)

    file_name = "result_" + str(data) + "_" + str(normalized) + "_" + \
        str(distance) + "_" + str(k) + ".txt"

    fout = open("results/" + file_name, "w")

    end_time = time.time()

    execution_time = end_time - start_time

    fout.write('Accuracy: ' + str(neigh.score(X_test, y_test)))
    fout.write('\n\nConfusion Matrix: \n\n' +
               str(confusion_matrix(y_test, y_pred)))
    fout.write('\n\nClassification Report: \n\n' + str(classification_report(
        y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
    fout.write('\nExecution Time: ' + str(execution_time))


if __name__ == "__main__":
    start_time = time.time()
    json_file = open('experiments.json')
    experiments = json.load(json_file)
    for experiment in experiments:
        print(experiment['data'])
        run(experiment['data'], experiment['normalized'],
            experiment['distance'], experiment['k'])
