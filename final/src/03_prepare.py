import lib.utils as utils
import lib.normalizer as normalizer
import lib.tabulation as tabulation
from sklearn.datasets import load_svmlight_file

FILE_SVMLIGHT_TRAIN_INPUT = 'data/train/credit_sample.svmlight'
FILE_SVMLIGHT_VALIDATION_INPUT = 'data/validation/credit_sample.svmlight'
FILE_SVMLIGHT_TEST_INPUT = 'data/test/credit_sample.svmlight'

FILE_SVMLIGHT_TRAIN_OUTPUT = 'data/train/credit_sample_normalized.svmlight'
FILE_SVMLIGHT_VALIDATION_OUTPUT = 'data/validation/credit_sample_normalized.svmlight'
FILE_SVMLIGHT_TEST_OUTPUT = 'data/test/credit_sample_normalized.svmlight'

FILE_RESULT_NORMALIZE = 'result/normalize/tabulation.csv'

start_time = 0

if __name__ == "__main__":
    header = ['Type', 'Time']
    tabulation_writer = tabulation.get_tabulation(
        FILE_RESULT_NORMALIZE, header)

    start_time = utils.get_time()

    print(f'Normalizing train file on {FILE_SVMLIGHT_TRAIN_INPUT}.')
    x, y = normalizer.normalize_data(FILE_SVMLIGHT_TRAIN_INPUT)
    diff_time = utils.get_time_diff(start_time)
    print(f'Normalizing Executed in {diff_time} seconds.')
    tabulation.save_tabulation_row(tabulation_writer, ['Train', diff_time])
    normalizer.save_normalized_data(x, y, FILE_SVMLIGHT_TRAIN_OUTPUT)

    start_time = utils.get_time()
    print(f'Normalizing train file on {FILE_SVMLIGHT_VALIDATION_INPUT}.')
    x, y = normalizer.normalize_data(FILE_SVMLIGHT_VALIDATION_INPUT)
    diff_time = utils.get_time_diff(start_time)
    print(f'Normalizing Executed in {diff_time} seconds.')
    tabulation.save_tabulation_row(
        tabulation_writer, ['Validation', diff_time])
    normalizer.save_normalized_data(x, y, FILE_SVMLIGHT_VALIDATION_OUTPUT)

    start_time = utils.get_time()
    print(f'Normalizing train file on {FILE_SVMLIGHT_TEST_INPUT}.')
    x, y = normalizer.normalize_data(FILE_SVMLIGHT_TEST_INPUT)
    diff_time = utils.get_time_diff(start_time)
    print(f'Normalizing Executed in {diff_time} seconds.')
    tabulation.save_tabulation_row(tabulation_writer, ['Test', diff_time])
    normalizer.save_normalized_data(x, y, FILE_SVMLIGHT_TEST_OUTPUT)
