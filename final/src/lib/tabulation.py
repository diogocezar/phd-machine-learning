import csv


def get_tabulation(file, header, delimiter=','):
    tabulation_file = open(file, mode='w')
    tabulation_writer = csv.writer(
        tabulation_file, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    tabulation_writer.writerow(header)
    return [tabulation_writer, tabulation_file]


def save_tabulation_row(tabulation_writer, row):
    tabulation_writer.writerow(row)
