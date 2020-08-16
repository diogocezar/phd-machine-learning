from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn import svm
import numpy
from . import utils


def classify_generic(classificator, x_train, y_train, x_test, y_test, start_time):
    print(f'\tRunning: {classificator}')
    print(f'\tStarting fit ({utils.get_time_diff(start_time)}s)')
    classificator.fit(x_train, y_train)
    print(f'\tFinishing fit ({utils.get_time_diff(start_time)}s)')
    print(f'\tStarting predict ({utils.get_time_diff(start_time)}s)')
    predict = classificator.predict(x_test)
    print(f'\tFinishing predict ({utils.get_time_diff(start_time)}s)')
    print(f'\tStarting f1_score ({utils.get_time_diff(start_time)}s)')
    result_f1_score = utils.round_float(f1_score(
        y_test, predict, labels=numpy.unique(predict), average='weighted'))
    print(f'\tFinishing f1_score ({utils.get_time_diff(start_time)}s)')
    print(f'\tStarting accuracy ({utils.get_time_diff(start_time)}s)')
    result_accuracy = utils.round_float(classificator.score(x_test, y_test))
    print(f'\tFinishing accuracy ({utils.get_time_diff(start_time)}s)')
    print(f'\tStarting conf_mat ({utils.get_time_diff(start_time)}s)')
    result_conf_mat = confusion_matrix(y_test, predict)
    print(f'\tFinishing conf_mat ({utils.get_time_diff(start_time)}s)\n')
    result_time = utils.round_float(utils.get_time_diff(start_time))
    return[result_f1_score, result_accuracy, result_conf_mat, result_time]


def classify_svm(start_time, x_train, y_train, x_test, y_test):
    classificator = svm.SVC(kernel='linear')
    return classify_generic(classificator, x_train, y_train, x_test, y_test, start_time)


def classify_knn(start_time, x_train, y_train, x_test, y_test):
    classificator = KNeighborsClassifier(n_neighbors=5)
    return classify_generic(classificator, x_train, y_train, x_test, y_test, start_time)


def classify_naive_bayes(start_time, x_train, y_train, x_test, y_test):
    classificator = GaussianNB()
    return classify_generic(classificator, x_train, y_train, x_test, y_test, start_time)


def classify_lda(start_time, x_train, y_train, x_test, y_test):
    classificator = LinearDiscriminantAnalysis()
    return classify_generic(classificator, x_train, y_train, x_test, y_test, start_time)


def classify_logistic_regression(start_time, x_train, y_train, x_test, y_test):
    classificator = LogisticRegression(max_iter=500)
    return classify_generic(classificator, x_train, y_train, x_test, y_test, start_time)


def classify_perceptron(start_time, x_train, y_train, x_test, y_test):
    classificator = Perceptron()
    return classify_generic(classificator, x_train, y_train, x_test, y_test, start_time)
