from keras.models import Sequential, load_model
import cv2
import math
import numpy as np

_MODEL_FILENAME = 'models/model_road_detector.h5'


class ConvDetector:
    model = Sequential()
    resolution = 100  # pixels
    line_width = 2
    true_color = (0, 255, 0)
    false_color = (0, 0, 255)

    def __init__(self, modelFile):
        self.model = load_model(modelFile)

    def predict(self, _input):
        assert(_input.ndim == 4
               and _input.shape[1] == self.resolution
               and _input.shape[2] == self.resolution)
        return self.model.predict(_input)

    def process(self, _img):
        #cv2.imshow('Input', _img)
        (_height, _width, _) = _img.shape
        max_x = math.floor(_width / self.resolution)
        max_y = math.floor(_height / self.resolution)
        inputs = []
        for x in range(0, max_x):
            for y in range(0, max_y):
                inputs.append(_img[self.resolution * y:self.resolution * (y + 1),
                                   self.resolution * x:self.resolution * (x + 1)]
                              )
        inputs = np.array(inputs)

        prediction = self.predict(inputs)
        assert len(prediction) == max_x * max_y
        _overlay = _img.copy()
        _output = _img.copy()

        for x in range(0, max_x):
            for y in range(0, max_y):
                _index = (x*max_y) + y
                _color = self.true_color if prediction[_index] == 1 else self.false_color
                cv2.rectangle(_overlay,
                              (self.resolution * x, self.resolution * y),
                              (self.resolution * (x+1), self.resolution * (y + 1)),
                              _color,
                              -1)
        alpha = 0.2
        cv2.addWeighted(_overlay, alpha, _output, 1 - alpha,
                        0, _output)
        #cv2.imshow('Processed', _output)
        return _output



# Self-testing, comment if not needed
# data = cv2.imread('road2/road6_12.jpg')
# detector = ConvDetector(_MODEL_FILENAME)
# detector.process(data)



