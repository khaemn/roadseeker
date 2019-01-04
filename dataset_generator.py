import numpy as np
from PIL import Image
import os
import cv2

'''
    This script takes 2 folders with images and forms a combined image for each pair
    also performing scaling to given shape.
    By default, VAERoader takes 320*180 pixel image as input and provides the same size
    output.
    Heatmapped images usually are about 1280*720 (full-hd camera) images, heatmapped by OCV_RND project.
'''

_X_INPUT_DIR = 'heatmapping/heatmap_src'
_Y_INPUT_DIR = 'heatmapping/heatmap_out'
_DATASET_DIR = 'dataset'
_TRAIN_DIR = os.path.join(_DATASET_DIR, 'train')
_X_TRAIN_DIR = os.path.join(_TRAIN_DIR, 'X')
_Y_TRAIN_DIR = os.path.join(_TRAIN_DIR, 'Y')

_ATTACH_Y_TO_X = True

# Usual resolutuion of HD cam is 1280*720, we use /4 resolution here to save performance
_X_WIDTH = 160  # 320
_X_HEIGHT = 90  # 180

# I think that road heatmap resolution of R/20 will be enough
# _Y_WIDTH = 64
# _Y_HEIGHT = 36
_Y_WIDTH = _X_WIDTH
_Y_HEIGHT = _X_HEIGHT


def generate_dataset(resolution=(_X_WIDTH, _X_HEIGHT)):
    x_images = []
    y_images = []

    (gen_w, gen_h) = resolution

    # Scan both dirs
    for root_back, dirs_back, files_back in os.walk(_X_INPUT_DIR):
        for _file in files_back:
            x_images.append(_file)

    for root_back, dirs_back, files_back in os.walk(_Y_INPUT_DIR):
        for _file in files_back:
            y_images.append(_file)

    assert len(x_images) == len(y_images)

    total_files = len(x_images)
    iteration = 1

    for filename in x_images:
        print('Processing file', iteration, 'of', total_files)
        iteration += 1
        x_img = cv2.imread(os.path.join(_X_INPUT_DIR, filename))
        x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)

        y_img = cv2.imread(os.path.join(_Y_INPUT_DIR, filename))
        y_img = cv2.cvtColor(y_img, cv2.COLOR_BGR2RGB)

        assert x_img.shape == y_img.shape
        (height, width, depth) = x_img.shape

        x_resized = cv2.resize(x_img, (gen_w, gen_h), interpolation=cv2.INTER_LINEAR)
        y_resized = cv2.resize(y_img, (gen_w, gen_h), interpolation=cv2.INTER_LINEAR)

        if _ATTACH_Y_TO_X:
            # Concatenating input with expected output
            output = np.zeros((gen_h, gen_w*2, depth), dtype='uint8')
            output[:, :_X_WIDTH] = x_resized
            output[:, _X_WIDTH:] = y_resized

            output = Image.fromarray(output)
            output_path = os.path.join(_X_TRAIN_DIR, filename)
            output.save(output_path)
        else:
            x_out, y_out = Image.fromarray(x_resized), Image.fromarray(y_resized)
            x_out.save(os.path.join(_X_TRAIN_DIR, filename))
            y_out.save(os.path.join(_Y_TRAIN_DIR, filename))


if __name__ == '__main__':
    generate_dataset()