from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file


def load_svm_file(input_file):
    x, y = load_svmlight_file(input_file)
    return [x, y]


def dump_svm_file(x, y, output_file):
    dump_svmlight_file(x, y, output_file)
