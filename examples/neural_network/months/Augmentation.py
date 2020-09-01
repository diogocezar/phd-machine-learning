from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import glob
import os
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove

drive_path = './'
aug_path = drive_path + "data-aug"
train_file_aug = drive_path + 'train-aug.txt'


def replace(file_path, pattern):
    # Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh, 'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                if pattern in line:
                    continue
                new_file.write(line)
    # Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    # Remove original file
    remove(file_path)
    # Move new file
    move(abs_path, file_path)


def save_to_aug(label, subdir):
    # * means all if need specific format then *.csv
    list_of_files = glob.glob(aug_path + "/" + subdir + "/*")
    if len(list_of_files) > 0:
        latest_file = max(list_of_files, key=os.path.getctime)
        replace(train_file_aug, latest_file.replace(aug_path + "/", ""))
        arq_aug = open(train_file_aug, "a+")
        arq_aug.write(latest_file.replace(
            aug_path + "/", "") + " " + label + "\n")
        arq_aug.close()


def flip_rotation_brightness_zoom(path, zoom=[0.5, 1.0], brightness=[0.2, 1.0], rotation=90, flip_horizontal=False,
                                  flip_vertical=False, subdir="all"):
    path, label = path.split(' ')
    path = drive_path + 'data/' + path
    img = load_img(path)
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(zoom_range=zoom, brightness_range=brightness, rotation_range=rotation,
                                 horizontal_flip=flip_horizontal, vertical_flip=flip_vertical)
    # prepare iterator
    it = datagen.flow(samples, save_to_dir=aug_path +
                      "/" + subdir + "/", batch_size=1)
    save_to_aug(label, subdir)
    # generate samples and plot
    for i in range(1):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    # pyplot.show()


def random_zoom(path, zoom=[0.5, 1.0], subdir="zoom"):
    path, label = path.split(' ')
    path = drive_path + 'data/' + path
    img = load_img(path)
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(zoom_range=[0.5, 1.0])
    # prepare iterator
    it = datagen.flow(samples, save_to_dir=aug_path +
                      "/" + subdir + "/", batch_size=1)
    save_to_aug(label, subdir)
    # generate samples and plot
    for i in range(1):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    # pyplot.show()


def random_brightness(path, brightness=[0.2, 1.0], subdir="brightness"):
    path, label = path.split(' ')
    path = drive_path + 'data/' + path
    # load the image
    img = load_img(path)
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(brightness_range=brightness)
    # prepare iterator
    it = datagen.flow(samples, save_to_dir=aug_path +
                      "/" + subdir + "/", batch_size=1)
    save_to_aug(label, subdir)
    # generate samples and plot
    for i in range(1):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    # pyplot.show()


def random_rotation(path, rotation=90, subdir="rotation"):
    path, label = path.split(' ')
    path = drive_path + 'data/' + path
    img = load_img(path)
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(rotation_range=rotation)
    # prepare iterator
    it = datagen.flow(samples, save_to_dir=aug_path +
                      "/" + subdir + "/", batch_size=1)
    save_to_aug(label, subdir)
    # generate samples and plot
    for i in range(1):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    # pyplot.show()


def horizontal_vertical_flip(path, flip_horizontal=False, flip_vertical=False, subdir="flip"):
    path, label = path.split(' ')
    path = drive_path + 'data/' + path
    # load the image
    img = load_img(path)
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(
        horizontal_flip=flip_horizontal, vertical_flip=flip_vertical)
    # prepare iterator
    it = datagen.flow(samples, save_to_dir=aug_path +
                      "/" + subdir + "/", batch_size=1)
    save_to_aug(label, subdir)
    # generate samples and plot
    for i in range(1):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    # pyplot.show()


def horizontal_vertical_shift(path, size=0.5, bool_width=True, subdir="shift"):
    path, label = path.split(' ')
    path = drive_path + 'data/' + path
    # load the image
    img = load_img(path)
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    if bool_width:
        datagen = ImageDataGenerator(width_shift_range=size)
    else:
        datagen = ImageDataGenerator(height_shift_range=size)
    # prepare iterator
    it = datagen.flow(samples, save_to_dir=aug_path +
                      "/" + subdir + "/", batch_size=1)
    save_to_aug(label, subdir)
    # generate samples and plot
    for i in range(1):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    # pyplot.show()


# Train and Test files
train_file = drive_path + 'train.txt'

arq = open(train_file, 'r')
texto = arq.read()
train_paths = texto.split('\n')

train_paths.remove('')  # Remove empty lines
train_paths.sort()

for image_path in train_paths:
    print(image_path)
    horizontal_vertical_shift(image_path, bool_width=True)
    horizontal_vertical_shift(image_path, bool_width=False)
    horizontal_vertical_flip(image_path, flip_horizontal=True)
    horizontal_vertical_flip(image_path, flip_vertical=True)
    random_rotation(image_path)
    random_rotation(image_path, rotation=45)
    random_brightness(image_path)
    random_brightness(image_path, brightness=[0, 0.2])
    random_zoom(image_path)
    random_zoom(image_path, zoom=[0, 0.5])
    flip_rotation_brightness_zoom(image_path)
    flip_rotation_brightness_zoom(image_path, zoom=[0.1, 0.5], brightness=[
                                  0.1, 0.5], flip_horizontal=True)
    flip_rotation_brightness_zoom(image_path, zoom=[0.1, 0.5], brightness=[
                                  0.1, 0.5], flip_vertical=True)
    flip_rotation_brightness_zoom(image_path, zoom=[0.1, 0.5], brightness=[
                                  0.1, 0.5], rotation=180)
    flip_rotation_brightness_zoom(image_path, zoom=[0.1, 0.5], brightness=[
                                  0.1, 0.5], rotation=180, flip_vertical=True)
    flip_rotation_brightness_zoom(image_path, zoom=[0.1, 0.5], brightness=[
                                  0.1, 0.5], rotation=180, flip_horizontal=True)
    flip_rotation_brightness_zoom(image_path, zoom=[0, 0.8], brightness=[0, 0.8], rotation=45,
                                  flip_vertical=True)
    flip_rotation_brightness_zoom(image_path, zoom=[0, 0.8], brightness=[0, 0.8], rotation=45,
                                  flip_vertical=True)
    flip_rotation_brightness_zoom(image_path, zoom=[0.1, 0.2], brightness=[
                                  0.1, 0.2], rotation=180)
    flip_rotation_brightness_zoom(
        image_path, zoom=[0.1, 0.2], brightness=[0.1, 0.2], rotation=45)
    flip_rotation_brightness_zoom(
        image_path, zoom=[0.9, 1], brightness=[0.9, 1], rotation=180)
    flip_rotation_brightness_zoom(
        image_path, zoom=[0.9, 1], brightness=[0.9, 1], rotation=45)
