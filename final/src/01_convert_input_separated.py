import csv
import lib.utils as utils
import collections

FILE_CSV_INPUT = 'data/credit.csv'
FILE_SVMLIGHT_OUTPUT_0 = 'data/credit_0.svmlight'
FILE_SVMLIGHT_OUTPUT_1 = 'data/credit_1.svmlight'

start_time = 0


def convert_input(file_csv_input, file_svmlight_output_0, file_svmlight_output_1):
    output_file_0 = open(file_svmlight_output_0, mode='w')
    output_file_1 = open(file_svmlight_output_1, mode='w')
    num_lines_output_file = utils.get_num_rows(file_csv_input) - 1
    with open(file_csv_input, mode='r') as input_file:
        csv_reader = csv.DictReader(input_file)
        line_count = 0
        line_count_0 = 0
        line_count_1 = 0
        for row in csv_reader:
            line = ""
            line = row["Y"] + " "
            values_0 = {}
            values_1 = {}
            for v in row:
                if v != "ID" and v != "Y":
                    index = int(v.replace("v", ""))
                    value = row[v]
                    if value == "NA":
                        value = 0
                    if row["Y"] == "0":
                        values_0[index] = value
                    else:
                        values_1[index] = value
            if row["Y"] == "0":
                values_0 = collections.OrderedDict(sorted(values_0.items()))
                for v, k in values_0.items():
                    line += str(v) + ":" + str(k) + " "
                line_count_0 += 1
                output_file_0.write(line)
                output_file_0.write("\n")
            else:
                if line_count_0 > line_count_1:
                    values_1 = collections.OrderedDict(
                        sorted(values_1.items()))
                    for v, k in values_1.items():
                        line += str(v) + ":" + str(k) + " "
                    line_count_1 += 1
                    output_file_1.write(line)
                    output_file_1.write("\n")
            line_count += 1
            print(f'Line {line_count} of {num_lines_output_file}')
        print(f'Processed {line_count} of {num_lines_output_file} lines.')
        print(f'0 -> {line_count_0}')
        print(f'1 -> {line_count_1}')


if __name__ == "__main__":
    start_time = utils.get_time()
    convert_input(FILE_CSV_INPUT, FILE_SVMLIGHT_OUTPUT_0,
                  FILE_SVMLIGHT_OUTPUT_1)
    print(f'Executed in {utils.get_time_diff(start_time)} seconds.')
