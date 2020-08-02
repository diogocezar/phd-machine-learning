import sys
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
from sklearn.metrics import f1_score
import pylab as pl
import seaborn as sns
import time
import json
import csv


def run(data, normalized, distance, k):
    print("# Analysing experiment with > data=", data, " normalized=",
          normalized, " distance=", distance, " k=", k)
    X_data, y_data = load_svmlight_file("features/results/" + data + ".txt")
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

    fout = open("experiments/results/" + file_name, "w")

    end_time = time.time()

    execution_time = end_time - start_time

    f1s = str(f1_score(y_test, y_pred, labels=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9], average='weighted'))

    accuracy = str(neigh.score(X_test, y_test))

    fout.write('Accuracy: ' + accuracy)
    fout.write('\n\nConfusion Matrix: \n\n' +
               str(confusion_matrix(y_test, y_pred)))
    fout.write('\n\nClassification Report: \n\n' + str(classification_report(
        y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
    fout.write('\n\nF1Score:' + f1s)
    fout.write('\nExecution Time: ' + str(execution_time))

    print("# Resulted > accuracy=", accuracy, " execution_time=",
          str(execution_time), "fscore=", f1_score)

    return accuracy, f1s, execution_time


if __name__ == "__main__":
    start_time = time.time()
    json_file_experiments = open('experiments/index.json')
    csv_file_tabulation = open('experiments/tabulation/index.csv', mode='w')
    experiments = json.load(json_file_experiments)

    tabulation_writer = csv.writer(
        csv_file_tabulation, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    tabulation_writer.writerow(
        ['Experiment', 'Normalized', 'Distance', 'K', 'Accuracy', 'F1Score', 'Execution Time'])

    for experiment in experiments:
        accuracy, f1s, execution_time = run(experiment['data'], experiment['normalized'],
                                            experiment['distance'], experiment['k'])
        tabulation_writer.writerow([experiment['data'], experiment['normalized'],
                                    experiment['distance'], experiment['k'], str(accuracy), str(f1s), str(execution_time)])

    json_file_experiments.close()
    csv_file_tabulation.close()
