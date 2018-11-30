import numpy as np
import csv
import PIL
import numpy
import random
from PIL import Image
from os import listdir
import os

# TODO: parse command line args
_INPUT_DIR = './road2'
_OUTPUT_DIR_0 = './train/generated/empty'
_OUTPUT_DIR_1 = './train/generated/road'

_ALLOWED_IMAGE_EXTENSIONS = ['.png', '.jpg', 'jpeg']
_MODEL_EXTENSION = '.txt'

images = []
models = []

for root_back, dirs_back, files_back in os.walk(_INPUT_DIR):
    for _file in files_back:
        # TODO: use list of extensions here
        if _file.endswith(_ALLOWED_IMAGE_EXTENSIONS[1]):
            images.append(_file)
        if(_file.endswith(_MODEL_EXTENSION)):
            models.append(_file)

models.sort()
images.sort()

assert len(models) == len(images)
total_files = len(models)

for i in range(0, total_files):

    model_file = open(os.path.join(root_back, models[i]), 'r')
    csv_reader = csv.reader(model_file, delimiter=',')
    model_data = []
    for row in csv_reader:
        model_data.append(row)

    (pixel_grid_size, origin_width, origin_height) = model_data[0]
    model_data.remove(model_data[0])
    pixel_grid_size, origin_width, origin_height = int(pixel_grid_size), int(origin_width), int(origin_height)

    model = np.zeros((origin_width, origin_height))
    for record in model_data:
        (_x, _y, _state) = record
        _x, _y, _state = int(_x), int(_y), int(_state)
        model[_x, _y] = _state

    # TODO: parse model file
    img = Image.open(os.path.join(root_back, images[i]))
    try:
        data = np.asarray(img, dtype='uint8')
    except SystemError:
        data = np.asarray(img.getdata(), dtype='uint8')

    for x in range(0, origin_width - 1):
        for y in range (0, origin_height - 1):
            cropped = data[y * pixel_grid_size : (y+1) * pixel_grid_size,
                           x * pixel_grid_size : (x+1) * pixel_grid_size]

            if cropped.shape[0] != cropped.shape[1]:
                pass
            try:
                output = Image.fromarray(cropped, "RGB")
            except ValueError:
                pass

            contains_selection = (model[x, y] > 0)
            output_path = _OUTPUT_DIR_1 if contains_selection else _OUTPUT_DIR_0
            output.save(os.path.join(output_path, "_" + str(x) + "_" + str(y) + "_" + images[i]))
