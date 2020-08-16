import csv
from . import utils
import os


def get_tabulation(file, header, delimiter=','):
    tabulation_file = open(file, mode='w')
    tabulation_writer = csv.writer(
        tabulation_file, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    tabulation_writer.writerow(header)
    return [tabulation_writer, tabulation_file]


def save_tabulation_row(tabulation_writer, row):
    tabulation_writer.writerow(row)


def save_tabulation_conf_mat(path, classifier, result_conf_mat, folder):
    mk_folder = path + "/" + folder
    if not os.path.exists(mk_folder):
        os.mkdir(mk_folder)
    tabulation_csv_file = open(
        mk_folder + "/" + str(classifier) + '.csv', mode='w')
    tabulation_writer = csv.writer(
        tabulation_csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    tabulation_writer.writerows(result_conf_mat)
