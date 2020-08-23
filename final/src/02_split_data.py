import lib.split as split
import lib.utils as utils
import lib.tabulation as tabulation

FILE_SVMLIGHT_INPUT = 'data/credit.svmlight'

FILE_SVMLIGHT_TRAIN_OUTPUT = 'data/train/credit.svmlight'
FILE_SVMLIGHT_VALIDATION_OUTPUT = 'data/validation/credit.svmlight'
FILE_SVMLIGHT_TEST_OUTPUT = 'data/test/credit.svmlight'

FILE_RESULT_SPLIT = 'result/split/tabulation.csv'

start_time = 0


def save_split_results(x_train, x_validation, x_test, y_train, y_validation, y_test):
    header = ['Type', 'Length']
    tabulation_writer, tabulation_file = tabulation.get_tabulation(
        FILE_RESULT_SPLIT, header)
    print(f'Saving train file on {FILE_SVMLIGHT_TRAIN_OUTPUT}.')
    split.save_splited_data(x_train, y_train, FILE_SVMLIGHT_TRAIN_OUTPUT)
    tabulation.save_tabulation_row(
        tabulation_writer, ['Train', str(len(x_train))])
    print(f'Saving validation file on {FILE_SVMLIGHT_VALIDATION_OUTPUT}.')
    split.save_splited_data(x_validation, y_validation,
                            FILE_SVMLIGHT_VALIDATION_OUTPUT)
    tabulation.save_tabulation_row(
        tabulation_writer, ['Validation', len(x_validation)])
    print(f'Saving train file on {FILE_SVMLIGHT_TEST_OUTPUT}.')
    split.save_splited_data(x_test, y_test, FILE_SVMLIGHT_TEST_OUTPUT)
    tabulation.save_tabulation_row(
        tabulation_writer, ['Test', len(x_test)])

    tabulation_file.close()


if __name__ == "__main__":
    start_time = utils.get_time()
    x_train, x_validation, x_test, y_train, y_validation, y_test = split.split_data(
        FILE_SVMLIGHT_INPUT)
    print(f'Split Executed in {utils.get_time_diff(start_time)} seconds.')
    save_split_results(x_train, x_validation, x_test,
                       y_train, y_validation, y_test)
