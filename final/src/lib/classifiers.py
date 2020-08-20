from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy
from . import utils
from . import roc
from . import conf_mat


def classify_generic(name, folder, classificator, x_train, y_train, x_test, y_test, start_time):
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
    result_accuracy = utils.round_float(accuracy_score(y_test, predict))
    print(f'\tFinishing accuracy ({utils.get_time_diff(start_time)}s)')

    print(f'\tStarting precision ({utils.get_time_diff(start_time)}s)')
    result_precision = utils.round_float(precision_score(
        y_test, predict))
    print(f'\tFinishing precision ({utils.get_time_diff(start_time)}s)')

    print(f'\tStarting recall ({utils.get_time_diff(start_time)}s)')
    result_recall = utils.round_float(recall_score(y_test, predict))
    print(f'\tFinishing recall ({utils.get_time_diff(start_time)}s)')

    print(f'\tStarting conf_mat ({utils.get_time_diff(start_time)}s)')
    result_conf_mat = confusion_matrix(y_test, predict.round())
    print(f'\tFinishing conf_mat ({utils.get_time_diff(start_time)}s)\n')

    print(f'\tStarting saving roc graph ({utils.get_time_diff(start_time)}s)')
    roc.save_roc(folder, name, y_test, predict)
    print(
        f'\tFinishing saving roc graph ({utils.get_time_diff(start_time)}s)\n')

    print(f'\tStarting saving mat conf ({utils.get_time_diff(start_time)}s)')
    conf_mat.save_conf_mat(folder, classificator, x_test, y_test)
    print(
        f'\tFinishing saving mat conf ({utils.get_time_diff(start_time)}s)\n')

    result_time = utils.round_float(utils.get_time_diff(start_time))

    return[result_f1_score, result_accuracy, result_precision, result_recall, result_conf_mat, result_time]


def classify_svm(folder, start_time, x_train, y_train, x_test, y_test):
    classificator = svm.LinearSVC()
    return classify_generic("svm", folder, classificator, x_train, y_train, x_test, y_test, start_time)


def classify_knn(folder, start_time, x_train, y_train, x_test, y_test):
    classificator = KNeighborsClassifier(n_neighbors=5)
    return classify_generic("knn", folder, classificator, x_train, y_train, x_test, y_test, start_time)


def classify_naive_bayes(folder, start_time, x_train, y_train, x_test, y_test):
    classificator = GaussianNB()
    return classify_generic("naive_bayes", folder, classificator, x_train, y_train, x_test, y_test, start_time)


def classify_lda(folder, start_time, x_train, y_train, x_test, y_test):
    classificator = LinearDiscriminantAnalysis()
    return classify_generic("lda", folder, classificator, x_train, y_train, x_test, y_test, start_time)


def classify_logistic_regression(folder, start_time, x_train, y_train, x_test, y_test):
    classificator = LogisticRegression(max_iter=500)
    return classify_generic("logistic_regression", folder, classificator, x_train, y_train, x_test, y_test, start_time)


def classify_perceptron(folder, start_time, x_train, y_train, x_test, y_test):
    classificator = Perceptron()
    return classify_generic("perceptron", folder, classificator, x_train, y_train, x_test, y_test, start_time)


def classify_tree(folder, start_time, x_train, y_train, x_test, y_test):
    classificator = DecisionTreeClassifier(criterion="entropy", max_depth=7)
    return classify_generic("tree", folder, classificator, x_train, y_train, x_test, y_test, start_time)


def classify_mlp(folder, start_time, x_train, y_train, x_test, y_test):
    # normalizar com z-score
    classificator = MLPClassifier(random_state=1, max_iter=200)
    return classify_generic("mlp", folder, classificator, x_train, y_train, x_test, y_test, start_time)
