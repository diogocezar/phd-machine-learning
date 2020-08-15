import csv
import lib.util as util
import collections

FILE_CSV_INPUT = 'data/credit_sample.csv'
FILE_SVMLIGHT_OUTPUT = 'data/credit_sample.svmlight'

start_time = 0


def convert_data(file_csv_input, file_svmlight_output):
    output_file = open(file_svmlight_output, mode='w')
    with open(file_csv_input, mode='r') as input_file:
        csv_reader = csv.DictReader(input_file)
        line_count = 0
        for row in csv_reader:
            line = ""
            line = row["Y"] + " "
            values = {}
            for v in row:
                if v != "ID" and v != "Y":
                    index = int(v.replace("v", ""))
                    value = row[v]
                    if value == "NA":
                        value = -1000000000
                    values[index] = value
            values = collections.OrderedDict(sorted(values.items()))
            for v, k in values.items():
                line += str(v) + ":" + str(k) + " "
            output_file.write(line)
            output_file.write("\n")
            line_count += 1
            print(f'Line {line_count} of 219984')
        print(f'Processed {line_count} lines.')


if __name__ == "__main__":
    start_time = util.get_time()
    convert_data(FILE_CSV_INPUT, FILE_SVMLIGHT_OUTPUT)
    print(f'Executed in {util.get_time_diff(start_time)} seconds.')
