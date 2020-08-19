import lib.orchestrator as orchestrator
import lib.tabulation as tabulation
import lib.svmlight_utils as svmlight_utils
import lib.classifiers as classifiers
import lib.utils as utils
import time

from sklearn.datasets import load_svmlight_file

FILE_ORCHESTRATOR = 'orchestrator/index.json'

start_time = 0


def print_result(classifier,
                 result_f1_score,
                 result_accuracy,
                 result_precision,
                 result_recall,
                 result_conf_mat,
                 result_time):
    print('Classifier: ' + classifier)
    print('F1Score: ' + str(result_f1_score))
    print('Accuracy: ' + str(result_accuracy))
    print('Precision: ' + str(result_precision))
    print('Recall: ' + str(result_recall))
    print('Confusion Matrix: \n' + str(result_conf_mat))
    print('Execution Time(s): ' + str(result_time))
    print('\n---\n')


def save_result(classifier,
                result_f1_score,
                result_accuracy,
                result_precision,
                result_recall,
                result_conf_mat,
                result_time,
                tabulation_writer,
                path_conf_mat,
                folder):
    print_result(classifier,
                 result_f1_score,
                 result_accuracy,
                 result_precision,
                 result_recall,
                 result_conf_mat,
                 result_time)
    tabulation.save_tabulation_conf_mat(
        path_conf_mat, classifier, result_conf_mat, folder)
    tabulation_writer.writerow(
        [classifier, result_f1_score, result_accuracy, result_precision, result_recall, result_time])


def run_orchestrator(configs, experiments, start_time, table_writer, folder):
    start_time = time.time()
    print('Starting loading files')
    x_train, y_train = load_svmlight_file(configs["svmlight_train_input"])
    x_test, y_test = load_svmlight_file(configs["svmlight_test_input"])
    x_train = x_train.toarray()
    x_test = x_test.toarray()
    print(f'Finishing loading files ({utils.get_time_diff(start_time)}s)')
    print('Starting experiments: \n')
    for experiment in experiments:
        classifier = str(experiment['classifier'])
        start_time = time.time()
        if classifier == "knn":
            result_f1_score, result_accuracy, result_precision, result_recall, result_conf_mat, result_time = classifiers.classify_knn(
                folder, start_time, x_train, y_train, x_test, y_test)
        if classifier == "naive_bayes":
            result_f1_score, result_accuracy, result_precision, result_recall, result_conf_mat, result_time = classifiers.classify_naive_bayes(
                folder, start_time, x_train, y_train, x_test, y_test)
        if classifier == "lda":
            result_f1_score, result_accuracy, result_precision, result_recall, result_conf_mat, result_time = classifiers.classify_lda(
                folder, start_time, x_train, y_train, x_test, y_test)
        if classifier == "logistic_regression":
            result_f1_score, result_accuracy, result_precision, result_recall, result_conf_mat, result_time = classifiers.classify_logistic_regression(
                folder, start_time, x_train, y_train, x_test, y_test)
        if classifier == "perceptron":
            result_f1_score, result_accuracy, result_precision, result_recall, result_conf_mat, result_time = classifiers.classify_perceptron(
                folder, start_time, x_train, y_train, x_test, y_test)
        if classifier == "svm":
            result_f1_score, result_accuracy, result_precision, result_recall, result_conf_mat, result_time = classifiers.classify_svm(
                folder, start_time, x_train, y_train, x_test, y_test)
        if classifier == "tree":
            result_f1_score, result_accuracy, result_precision, result_recall, result_conf_mat, result_time = classifiers.classify_tree(
                folder, start_time, x_train, y_train, x_test, y_test)
        save_result(classifier,
                    result_f1_score,
                    result_accuracy,
                    result_precision,
                    result_recall,
                    result_conf_mat,
                    result_time,
                    table_writer,
                    configs["result_conf_mat"],
                    folder)


if __name__ == "__main__":
    header = ['Classifier', 'F1Score', 'Accuracy',
              'Precision', 'Recall', 'Execution Time (s)']
    orchestrator = orchestrator.get_orchestrator(FILE_ORCHESTRATOR)
    configs = orchestrator["configs"]
    experiments = orchestrator["experiments"]
    folder = str(utils.round_float(utils.get_time()))
    save_tabulation = configs["result_classifiers"].replace(
        "{timestamp}", folder)
    tabulation_writer, tabulation_file = tabulation.get_tabulation(
        save_tabulation, header)
    run_orchestrator(configs, experiments, start_time,
                     tabulation_writer, folder)
    tabulation_file.close()
