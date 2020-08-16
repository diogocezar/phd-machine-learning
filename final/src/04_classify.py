import lib.orchestrator as orchestrator
import lib.tabulation as tabulation
import lib.svmlight_utils as svmlight_utils
import lib.classifiers as classifiers
import time
import numpy
from sklearn.datasets import load_svmlight_file

FILE_SVMLIGHT_TRAIN_INPUT = 'data/train/credit_sample.svmlight'
FILE_SVMLIGHT_VALIDATION_INPUT = 'data/validation/credit_sample.svmlight'
FILE_SVMLIGHT_TEST_INPUT = 'data/test/credit_sample.svmlight'

FILE_ORCHESTRATOR = 'orchestrator/index.json'

FILE_RESULT_CLASSIFIERS = 'result/classify/tabulation.csv'

start_time = 0


def print_result(result_f1_score,
                 result_accuracy,
                 result_conf_mat,
                 result_time):
    print('F1Score: ' + str(result_f1_score))
    print('Accuracy: ' + str(result_accuracy))
    print('Confusion Matrix: \n' + str(result_conf_mat))
    print('Execution Time(s): ' + str(result_time))


def run_orchestrator(orchestrator, start_time, table_writer):
    print('Starting orchestrator...')
    x_train, y_train = load_svmlight_file(FILE_SVMLIGHT_TRAIN_INPUT)
    x_test, y_test = load_svmlight_file(FILE_SVMLIGHT_TEST_INPUT)
    x_train = x_train.toarray()
    x_test = x_test.toarray()
    labels = numpy.arange(706)
    for experiment in orchestrator:
        classifier = str(experiment['classifier'])
        print('Classifier: ' + classifier)
        start_time = time.time()
        if classifier == "knn":
            result_f1_score, result_accuracy, result_conf_mat, result_time = classifiers.classify_knn(
                start_time, labels, x_train, y_train, x_test, y_test)
        print_result(result_f1_score,
                     result_accuracy,
                     result_conf_mat,
                     result_time)


if __name__ == "__main__":
    header = ['Classifier', 'F1Score', 'Accuracy', 'Execution Time (s)']
    orchestrator = orchestrator.get_orchestrator(FILE_ORCHESTRATOR)
    tabulation_writer, tabulation_file = tabulation.get_tabulation(
        FILE_RESULT_CLASSIFIERS, header)
    run_orchestrator(orchestrator, start_time, tabulation_writer)
    tabulation_file.close()
