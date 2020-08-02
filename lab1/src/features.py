import os
import numpy as np
import random
import statistics
import time
import json
import csv
from cv2 import cv2


def run(data, x, y, avg_x, avg_y, med_x, med_y, max_x, max_y):
    start_time = time.time()
    use_x = use_y = 0
    if isinstance(x, int):
        use_x = x
    else:
        use_x = {
            "avg_x": int(avg_x),
            "med_x": int(med_x),
            "max_x": int(max_x)
        }[x]
    if isinstance(y, int):
        use_y = y
    else:
        use_y = {
            "avg_y": int(avg_y),
            "med_y": int(med_y),
            "max_y": int(max_y)
        }[y]
    print('# Generating features using:', use_x, use_y)
    file_name = "features/results/features_" + \
        str(use_x) + "_" + str(use_y) + ".txt"
    fout = open(file_name, "w")
    load_images('source/data', fout, int(use_x), int(use_y))
    file_stats = os.stat(file_name)
    fout.close
    end_time = time.time()
    execution_time = (end_time - start_time)
    print('# Execution Time: ', str(execution_time))
    return use_x, use_y, file_stats.st_size, execution_time


def get_sizes(path_images):
    archives = os.listdir(path_images)
    images_x = []
    images_y = []
    print('# Extracting avg sizes...')
    for archive in archives:
        image = cv2.imread(path_images + '/' + archive, 0)
        image_y, image_x = image.shape
        images_x.append(image_x)
        images_y.append(image_y)
    return statistics.mean(images_x), statistics.mean(images_y), statistics.median(images_x), statistics.median(images_y), max(images_x), max(images_y)


def load_images(path_images, fout, X, Y):
    archives = os.listdir(path_images)
    images = []
    source_index = open('source/index.txt')
    lines = source_index.readlines()
    for line in lines:
        aux = line.split('/')[1]
        image_name = aux.split(' ')[0]
        label = line.split(' ')[1]
        label = label.split('\n')
        for archive in archives:
            if archive == image_name:
                image = cv2.imread(path_images + '/' + archive, 0)
                rawpixel(image, label[0], fout, X, Y)
    return images


def rawpixel(image, label, fout, X, Y):
    image = cv2.resize(image, (X, Y))
    fout.write(str(label) + " ")
    index = 0
    for i in range(Y):
        for j in range(X):
            if(image[i][j] > 128):
                v = 0
            else:
                v = 1
            fout.write(str(index)+":"+str(v)+" ")
            index = index+1
    fout.write("\n")


if __name__ == "__main__":
    json_file = open('features/index.json')
    csv_file_tabulation = open('features/tabulation/index.csv', mode='w')
    tabulation_writer = csv.writer(
        csv_file_tabulation, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    tabulation_writer.writerow(
        ['X_Source', 'Y_Source', 'X', 'Y', 'File Size', 'Execution Time'])
    features = json.load(json_file)
    avg_x, avg_y, med_x, med_y, max_x, max_y = get_sizes('source/data')
    for feature in features:
        use_x, use_y, file_size, execution_time = run(feature['data'], feature['x'],
                                                      feature['y'], avg_x, avg_y, med_x, med_y, max_x, max_y)
        tabulation_writer.writerow([feature['x'], feature['y'],
                                    use_x, use_y, str(file_size), str(execution_time)])
    json_file.close()
    csv_file_tabulation.close()
