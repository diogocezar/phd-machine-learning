import lib.split as split

if __name__ == "__main__":
    x_train, x_validation, x_test, y_train, y_validation, y_test = split.split_data(
        'data/credit_sample.svmlight')
    print('Train:')
    print(len(x_train))
    print(len(y_train))
    print('Validation:')
    print(len(x_validation))
    print(len(y_validation))
    print('Test:')
    print(len(x_test))
    print(len(y_test))
