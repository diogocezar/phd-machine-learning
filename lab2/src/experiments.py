import sys
import numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
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


def classify_generic(classificator, chunk, start_time):
    x_train, y_train = get_chunk_train_base(chunk)
    x_test, y_test = get_test_base()
    classificator.fit(x_train, y_train)
    predict = classificator.predict(x_test)
    result_f1_score = round_float(f1_score(
        y_test, predict, labels=labels, average='weighted'))
    result_accuracy = round_float(classificator.score(x_test, y_test))
    result_conf_mat = confusion_matrix(y_test, predict)
    result_time = round_float(get_time(start_time))
    return[result_f1_score, result_accuracy, result_conf_mat, result_time]


def classify_knn(chunk, start_time):
    classificator = KNeighborsClassifier()
    return classify_generic(classificator, chunk, start_time)


def classify_naive_bayes(chunk, start_time):
    classificator = GaussianNB()
    return classify_generic(classificator, chunk, start_time)


def classify_lda(chunk, start_time):
    classificator = LinearDiscriminantAnalysis()
    return classify_generic(classificator, chunk, start_time)


def classify_logistic_regression(chunk, start_time):
    classificator = LogisticRegression()
    return classify_generic(classificator, chunk, start_time)


def classify_perceptron(chunk, start_time):
    classificator = Perceptron()
    return classify_generic(classificator, chunk, start_time)


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
        ['Classifier', 'Start', 'Stop', 'Step', 'Chunk', 'F1Score', 'Accuracy', 'Execution Time (s)'])
    return [tabulation_writer, tabulation_csv_file]


def save_tabulation_conf_mat(classifier, i, result_conf_mat):
    tabulation_csv_file = open(
        'tabulation/conf_mat/' + str(classifier) + '_' + str(i) + '.csv', mode='w')
    tabulation_writer = csv.writer(
        tabulation_csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    tabulation_writer.writerows(result_conf_mat)


def print_result(result_f1_score,
                 result_accuracy,
                 result_conf_mat,
                 result_time):
    print('F1Score: ' + str(result_f1_score))
    print('Accuracy: ' + str(result_accuracy))
    print('Confusion Matrix: \n' + str(result_conf_mat))
    print('Execution Time(s): ' + str(result_time))


def save_result(classifier,
                start,
                stop,
                step,
                i,
                result_f1_score,
                result_accuracy,
                result_conf_mat,
                result_time,
                tabulation_writer):
    save_tabulation_conf_mat(classifier, i, result_conf_mat)
    tabulation_writer.writerow(
        [classifier, start, stop, step, i, result_f1_score, result_accuracy, result_time])


def get_time(start_time):
    end_time = time.time()
    return end_time - start_time


def round_float(value):
    return float("{:.3f}".format(value))


def run_orchestrator(orchestrator, start_time, table_writer):
    print('Starting orchestrator...')
    for experiment in orchestrator:
        print('Classifier: ' + str(experiment['classifier']))
        start = int(experiment['chunk_start'])
        stop = int(experiment['chunk_stop'])
        step = int(experiment['chunk_step'])
        classifier = str(experiment['classifier'])
        for i in range(start, stop, step):
            start_time = time.time()
            print('Chunk: ' + str(i))
            if classifier == "knn":
                result_f1_score, result_accuracy, result_conf_mat, result_time = classify_knn(
                    i, start_time)
            if classifier == "naive_bayes":
                result_f1_score, result_accuracy, result_conf_mat, result_time = classify_naive_bayes(
                    i, start_time)
            if classifier == "lda":
                result_f1_score, result_accuracy, result_conf_mat, result_time = classify_lda(
                    i, start_time)
            if classifier == "logistic_regression":
                result_f1_score, result_accuracy, result_conf_mat, result_time = classify_logistic_regression(
                    i, start_time)
            if classifier == "perceptron":
                result_f1_score, result_accuracy, result_conf_mat, result_time = classify_perceptron(
                    i, start_time)

            print_result(result_f1_score,
                         result_accuracy,
                         result_conf_mat,
                         result_time)
            save_result(classifier,
                        start,
                        stop,
                        step,
                        i,
                        result_f1_score,
                        result_accuracy,
                        result_conf_mat,
                        result_time,
                        table_writer)


if __name__ == "__main__":
    orchestrator = get_orchestrator()
    tabulation_writer, tabulation_csv_file = get_tabulation()
    run_orchestrator(orchestrator, start_time, tabulation_writer)
    tabulation_csv_file.close()
