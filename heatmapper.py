if __name__ != '__main__':
    quit()

from convroaddetector import *

_OUT_DIR = 'heatmapping/heatmap_out'
_SRC_DIR = 'heatmapping/heatmap_src'
_COMBINED_DIR = 'heatmapping/combined_out'

#_SRC_DIR = 'img'

make_heatmaps(input_path=_SRC_DIR,
              output_path=_OUT_DIR,
              combined_path=_COMBINED_DIR,
              oversampling_ratio=4,
              binary_threshold=170)
