from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file


def split_data(source_file):
    x, y = load_svmlight_file(source_file)
    x_train, x_test_50, y_train, y_test_50 = train_test_split(
        x, y, test_size=0.5, random_state=5)
    x_test, x_validation, y_test, y_validation = train_test_split(
        x_test_50, y_test_50, test_size=0.3, random_state=5)
    x_train = x_train.toarray()
    x_validation = x_validation.toarray()
    x_test = x_test.toarray()
    return [x_train, x_validation, x_test, y_train, y_validation, y_test]
