import numpy as np
import PIL
import numpy
import random
from PIL import Image
from os import listdir
import os

# TODO: parse command line args
_INPUT_DIR = './road2'
_OUTPUT_DIR_0 = './generated/0'
_OUTPUT_DIR_1 = './generated/1'

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


assert len(models) == len(images)
total_files = len(models)

for i in range(0, total_files):

    # Should be parsed from the Model file.
    pixel_grid_size = 100
    origin_width = 12  # 1280
    origin_height = 7  # 720

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
            output = Image.fromarray(cropped, "RGB")

            contains_selection = True  # TODO: read from model via xy coords
            output_path = _OUTPUT_DIR_1 if contains_selection else _OUTPUT_DIR_0
            output.save(os.path.join(output_path, "_" + str(x) + "_" + str(y) + "_" + images[i]))

    quit()
