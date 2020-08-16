def classify_generic(classificator, chunk, start_time):
    x_train, y_train = get_chunk_train_base(chunk)
    x_test, y_test = get_test_base()
    classificator.fit(x_train, y_train)
    predict = classificator.predict(x_test)
    result_f1_score = round_float(f1_score(
        y_test, predict, labels=labels, average='weighted'))
    result_accuracy = round_float(classificator.score(x_test, y_test))
    result_conf_mat = confusion_matrix(y_test, predict)
    result_time = round_float(get_time(start_time))
    return[result_f1_score, result_accuracy, result_conf_mat, result_time]


def classify_knn(chunk, start_time):
    classificator = KNeighborsClassifier()
    return classify_generic(classificator, chunk, start_time)


def classify_naive_bayes(chunk, start_time):
    classificator = GaussianNB()
    return classify_generic(classificator, chunk, start_time)


def classify_lda(chunk, start_time):
    classificator = LinearDiscriminantAnalysis()
    return classify_generic(classificator, chunk, start_time)


def classify_logistic_regression(chunk, start_time):
    classificator = LogisticRegression()
    return classify_generic(classificator, chunk, start_time)


def classify_perceptron(chunk, start_time):
    classificator = Perceptron()
    return classify_generic(classificator, chunk, start_time)
