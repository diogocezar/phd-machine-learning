import os
import numpy as np
import random
import statistics
import time
from cv2 import cv2


def get_sizes(path_images):
    archives = os.listdir(path_images)
    images_x = []
    images_y = []
    print('Extracting avg sizes...')
    for archive in archives:
        image = cv2.imread(path_images + '/' + archive, 0)
        image_x, image_y = image.shape
        images_x.append(image_x)
        images_y.append(image_y)
    return statistics.mean(images_x), statistics.mean(images_y), statistics.median(images_x), statistics.median(images_y), max(images_x), max(images_y)


def load_images(path_images, fout, X, Y):
    print('Loading images...')
    archives = os.listdir(path_images)
    images = []
    arq = open('source/index.txt')
    lines = arq.readlines()
    print('Extracting dummy features')
    print('Using: ', X, Y)
    for line in lines:
        aux = line.split('/')[1]
        image_name = aux.split(' ')[0]
        label = line.split(' ')[1]
        label = label.split('\n')
        for archive in archives:
            if archive == image_name:
                image = cv2.imread(path_images + '/' + archive, 0)
                rawpixel(image, label[0], fout, X, Y)
    print('Done. Take a look into features.txt')
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
    start_time = time.time()
    fout = open("features/features_10_20.txt", "w")
    avg_x, avg_y, med_x, med_y, max_x, max_y = get_sizes('source/images')
    base_x = 10
    base_y = 20
    images = load_images('source/images', fout, int(base_x), int(base_y))
    # images = load_images('source/images', fout, int(med_x), int(med_y))
    # images = load_images('source/images', fout, int(max_x), int(max_y))
    fout.close
    end_time = time.time()
    print('--- %s seconds ---' % (end_time - start_time))
