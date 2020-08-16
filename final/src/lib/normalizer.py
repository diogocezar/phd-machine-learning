from sklearn import preprocessing
from . import svmlight_utils


def fit_transform(base):
    scaler = preprocessing.MaxAbsScaler()
    return scaler.fit_transform(base)


def normalize_data(input_file):
    x, y = svmlight_utils.load_svm_file(input_file)
    x = fit_transform(x)
    return [x, y]


def save_normalized_data(x, y, output_file):
    svmlight_utils.dump_svm_file(x, y, output_file)
