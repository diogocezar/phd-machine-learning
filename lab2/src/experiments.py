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

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
start_time = 0


def get_chunk_train_base(size):
    x, y = load_svmlight_file("data/train.txt")
    x_array = x.toarray()
    return[x_array[0:size], y[0:size]]


def get_test_base():
    x, y = load_svmlight_file("data/test.txt")
    x_array = x.toarray()
    return[x_array, y]


def classify_knn(chunk):
    x_train, y_train = get_chunk_train_base(chunk)
    x_test, y_test = get_test_base()
    classificator = KNeighborsClassifier()
    classificator.fit(x_train, y_train)
    predict = classificator.predict(x_test)
    result_f1_score = f1_score(
        y_test, predict, labels=labels, average='weighted')
    result_accuracy = classificator.score(x_test, y_test)
    result_conf_mat = confusion_matrix(y_test, predict)
    result_time = get_time()
    return[result_f1_score, result_accuracy, result_conf_mat, result_time]


def get_orchestrator():
    orchestrator_json_file = open('orchestrator/index.json')
    orchestrator = json.load(orchestrator_json_file)
    orchestrator_json_file.close()
    return orchestrator


def get_tabulation():
    tabulation_csv_file = open('tabulation/index.csv', mode='w')
    tabulation_writer = csv.writer(
        tabulation_csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    tabulation_writer.writerow(
        ['Experiment', 'Normalized', 'Distance', 'K', 'Accuracy', 'F1Score', 'Execution Time'])
    return [tabulation_writer, tabulation_csv_file]


def get_time():
    end_time = time.time()
    return end_time - start_time


def run_each(orchestrator, start_time, table_writer):
    for experiment in orchestrator:
        print(experiment)


if __name__ == "__main__":
    start_time = time.time()
    print(classify_knn(10))
    # orchestrator = get_orchestrator()
    # tabulation_writer, tabulation_csv_file = get_tabulation()
    # #run_each(orchestrator, start_time, tabulation_writer)
    # print(get_chunk_train(1))
    # tabulation_csv_file.close()
