from sklearn.model_selection import train_test_split
from . import svmlight_utils


def split_data(input_file):
    x, y = svmlight_utils.load_svm_file(input_file)
    x_train, x_test_50, y_train, y_test_50 = train_test_split(
        x, y, test_size=0.5, random_state=5)
    x_test, x_validation, y_test, y_validation = train_test_split(
        x_test_50, y_test_50, test_size=0.3, random_state=5)
    x_train = x_train.toarray()
    x_validation = x_validation.toarray()
    x_test = x_test.toarray()
    return [x_train, x_validation, x_test, y_train, y_validation, y_test]


def save_splited_data(x, y, output_file):
    svmlight_utils.dump_svm_file(x, y, output_file)
