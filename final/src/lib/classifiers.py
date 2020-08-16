from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from . import utils


def classify_generic(classificator, labels, x_train, y_train, x_test, y_test, start_time):
    classificator.fit(x_train, y_train)
    predict = classificator.predict(x_test)
    result_f1_score = utils.round_float(f1_score(
        y_test, predict, labels=labels, average='weighted'))
    result_accuracy = utils.round_float(classificator.score(x_test, y_test))
    result_conf_mat = confusion_matrix(y_test, predict)
    result_time = utils.round_float(utils.get_time(start_time))
    return[result_f1_score, result_accuracy, result_conf_mat, result_time]


def classify_knn(start_time, labels, x_train, y_train, x_test, y_test):
    classificator = KNeighborsClassifier()
    return classify_generic(classificator, labels, x_train, y_train, x_test, y_test, start_time)


def classify_naive_bayes(start_time, labels, x_train, y_train, x_test, y_test):
    classificator = GaussianNB()
    return classify_generic(classificator, labels, x_train, y_train, x_test, y_test, start_time)


def classify_lda(start_time, labels, x_train, y_train, x_test, y_test):
    classificator = LinearDiscriminantAnalysis()
    return classify_generic(classificator, labels, x_train, y_train, x_test, y_test, start_time)


def classify_logistic_regression(start_time, labels, x_train, y_train, x_test, y_test):
    classificator = LogisticRegression()
    return classify_generic(classificator, labels, x_train, y_train, x_test, y_test, start_time)


def classify_perceptron(start_time, labels, x_train, y_train, x_test, y_test):
    classificator = Perceptron()
    return classify_generic(classificator, labels, x_train, y_train, x_test, y_test, start_time)
