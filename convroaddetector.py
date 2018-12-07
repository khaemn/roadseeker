from keras.models import Sequential, load_model
import cv2
import math
import numpy as np

_MODEL_FILENAME = 'models/model_road_detector.h5'
_CONVERT_TO_HSL = False

np.set_printoptions(precision=2, suppress=True)


class ConvDetector:
    model = Sequential()
    max_RGB = 255
    resolution = 100  # pixels
    line_width = 2
    true_color = (0, max_RGB, 0)
    false_color = (0, 0, max_RGB)
    threshold = 0.2

    def __init__(self, modelFile):
        self.model = load_model(modelFile)

    def predict(self, _input):
        assert(_input.ndim == 4
               and _input.shape[1] == self.resolution
               and _input.shape[2] == self.resolution)
        return self.model.predict(_input)

    def split_input(self, _img, x_offset=0, y_offset=0):
        (_height, _width, _) = _img.shape
        inputs = []
        max_x = round((_width  # - x_offset
                       ) / self.resolution)
        max_y = round((_height  # - x_offset
                       ) / self.resolution)
        x_limit = 0 if x_offset <= 0 else 1
        y_limit = 0 if y_offset <= 0 else 1
        #max_x -= x_limit
        #max_y -= y_limit
        if _height == self.resolution and _width == self.resolution:
            inputs.append(_img)
        else:
            for x in range(0, max_x):
                for y in range(0, max_y):
                    cropped = _img[self.resolution * y + y_offset:self.resolution * (y + 1) + y_offset,
                                  self.resolution * x + x_offset:self.resolution * (x + 1) + x_offset]
                    c_height, c_width, _ = cropped.shape
                    # If the cropped part is smaller than necessary, extend it with zeroes
                    if c_height < self.resolution or c_width < self.resolution:
                        corrected = np.zeros((self.resolution, self.resolution, 3))
                        corrected[:c_height, :c_width] = cropped
                        cropped = corrected

                    # print("\nX:%d, Y:%d, x_offset:%d, y_offset:%d, max_xL:%d, max_yL:%d"
                          # % (x, y, x_offset, y_offset, max_x - x_limit, max_y - y_limit), cropped.shape)
                    inputs.append(cropped)
        inputs = np.array(inputs, dtype='float32')
        inputs = inputs / self.max_RGB
        return max_x, max_y, inputs

    def process(self, _img, _offset=50):
        '''

        :param _img: np 2d array of RGB pixels
        :param _offset: offset both by Y and X in pixels (counting from top left to right bottom)
        :return:
        '''
        (_height, _width, _) = _img.shape

        max_x, max_y, inputs = self.split_input(_img, _offset)
        prediction = self.predict(inputs)
        print("Splitting from Processor:", max_x, max_y, len(prediction))
        #print(prediction)  # Print some predictions for debug
        # assert len(prediction) == max_x * max_y
        _overlay = _img.copy()
        if _CONVERT_TO_HSL:
            _img = cv2.cvtColor(_img, cv2.COLOR_HLS2RGB)
        else:
            _img = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR)

        for x in range(0, max_x):
            for y in range(0, max_y):
                _index = (x*max_y) + y
                yes = prediction[_index] > 0.5 + self.threshold
                no = prediction[_index] < 0.5 - self.threshold
                _color = self.true_color if yes else self.false_color if no else (0,0,0)
                cv2.rectangle(_overlay,
                              (self.resolution * x + _offset, self.resolution * y + _offset),
                              (self.resolution * (x+1) + _offset, self.resolution * (y + 1) + _offset),
                              _color,
                              -1)
                text = "YES" if yes else "NO" if no else "?"
                cv2.putText(_img, text, (self.resolution * x + 30 + _offset, self.resolution * y + 30 + _offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (self.max_RGB, self.max_RGB, self.max_RGB), 1, cv2.LINE_AA)
        alpha = 0.2
        cv2.addWeighted(_overlay, alpha, _img, 1 - alpha,
                        0, _img)
        if _width > 1300:
            _img = cv2.resize(_img, (int(_width / 2), int(_height / 2)))
        cv2.imshow('Processed', _img)
        return _img

    def heatmap(self, _img, oversampling_ratio=2, threshold=0.8):
        (_height, _width, _) = _img.shape

        heat_stride = int(self.resolution / oversampling_ratio)

        h_width = round(_width / self.resolution) * oversampling_ratio
        h_height = round(_height / self.resolution) * oversampling_ratio
        heatmap = np.zeros((h_height, h_width))
        _overlay = _img.copy()

        neuron_evaluations = 0
        # Filling the heatmap
        for x in range(0, oversampling_ratio):
            for y in range(0, oversampling_ratio):
                x_offset = x * heat_stride
                y_offset = y * heat_stride
                max_x, max_y, inputs = self.split_input(_img, x_offset, y_offset)
                prediction = self.predict(inputs)
                neuron_evaluations += len(inputs)
                print("Splitting from Overlapper:", max_x, max_y, len(prediction))
                # print(prediction)
                # Reshaping prediction to restore 2-d matrix
                prediction = prediction.reshape((max_x, max_y))
                prediction = prediction.transpose()
                # print(prediction)

                # Adding the predicted map to main heatmap. Each cell on the prediction should cover
                # *overlapping_ratio cells in the main heatmap.
                for h_x in range(0, max_x):
                    for h_y in range(0, max_y):
                        cell_value = prediction[h_y, h_x]
                        # a[2:4] += 5 https://stackoverflow.com/questions/32542689
                        center_y = h_y * oversampling_ratio + y
                        center_x = h_x * oversampling_ratio + x
                        cell_size_offset = oversampling_ratio
                        top_left_y = center_y  #  - _offset
                        top_left_x = center_x  # - _offset
                        bottom_right_y = min(h_height, top_left_y + cell_size_offset) + 1  # +1 because of NP range specific
                        bottom_right_x = min(h_width, top_left_x + cell_size_offset) + 1  # +1 because of NP range specific

                        #print("HX:%d, HY:%d, tlx:%d, tly:%d, cx:%d, cy:%d, brx:%d, bry:%d, added val:%1.1f"
                              #% (h_x, h_y, top_left_x, top_left_y, center_x, center_y,
                                 #bottom_right_x, bottom_right_y, cell_value))
                        heatmap[top_left_y : bottom_right_y,
                                top_left_x : bottom_right_x
                                ] += cell_value
                # impl rebound here
        print("Neuron network evaluations:", neuron_evaluations)
        # Visualizing the heatmap
        for x in range(0, h_width):
            for y in range(0, h_height):
                _heat_threshold = (oversampling_ratio) ** 2 * threshold
                is_road = heatmap[y, x] > _heat_threshold
                _color = self.true_color if is_road else self.false_color
                cv2.rectangle(_overlay,
                              (heat_stride * x, heat_stride * y),
                              (heat_stride * (x + 1), heat_stride * (y + 1)),
                              _color,
                              -1)
                text = "%2.1f" % heatmap[y, x]
                font_height = 1 / oversampling_ratio
                cv2.putText(_img, text, (heat_stride * x + 10, heat_stride * y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, font_height, (self.max_RGB, self.max_RGB, self.max_RGB), 1, cv2.LINE_AA)

        # print('Heatmap:', heatmap)

        if _CONVERT_TO_HSL:
            _img = cv2.cvtColor(_img, cv2.COLOR_HLS2RGB)
        else:
            _img = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR)
        alpha = 0.2
        cv2.addWeighted(_overlay, alpha, _img, 1 - alpha,
                        0, _img)
        if _width > 1300:
            _img = cv2.resize(_img, (int(_width / 2), int(_height / 2)))
        cv2.imshow('Heatmap', _img)
        return heatmap

def __test__(filenames):
    for filename in filenames:
        data = cv2.imread(filename)
        (height, width, _) = data.shape

        detector = ConvDetector(_MODEL_FILENAME)

        if _CONVERT_TO_HSL:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2HLS)
        else:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        #detector.process(data, _offset=0)
        oversampling = 10
        interpolation = cv2.INTER_LANCZOS4
        #interpolation = cv2.INTER_CUBIC
        heatmap = detector.heatmap(data, oversampling_ratio=oversampling, threshold=0.7)
        heatmap = heatmap / np.amax(heatmap) * 255
        heatmap = cv2.resize(heatmap, (int(width/2), int(height/2)), interpolation=interpolation)
        cv2.imshow("Raw map", heatmap / 100)
        _, threshed = cv2.threshold(heatmap, 170, 255, cv2.THRESH_BINARY)
        cv2.imshow("Thresholded", threshed)

        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    __test__([#'train/generated/empty/_0_5_road_x_17.png',
            #, 'train/generated/empty/_17_0_road_x_5.png'
            #, 'train/generated/road/_0_6_road_x_1.png'
            #, 'train/generated/road/_0_6_road_x_5.png'
            # 'img/lanes1.jpg'
            # 'img/lanes1_r.jpg'
            #'img/lanes9.jpg'
             'img/lanes10.jpg'
            #, 'img/lanes2.jpg'
            , 'img/lanes8.jpg'
            , 'img/lanes5.png'
            , 'img/lanes4.png'
            , 'img/lanes3.png'
            #, 'img/lanes6.jpg'
            ])

