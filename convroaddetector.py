from keras.models import Sequential, load_model
import cv2
import math
import numpy as np

_MODEL_FILENAME = 'models/model_road_detector.h5'
_CONVERT_TO_HSL = False


class ConvDetector:
    model = Sequential()
    resolution = 100  # pixels
    line_width = 2
    true_color = (0, 255, 0)
    false_color = (0, 0, 255)
    threshold = 0.1

    def __init__(self, modelFile):
        self.model = load_model(modelFile)

    def predict(self, _input):
        assert(_input.ndim == 4
               and _input.shape[1] == self.resolution
               and _input.shape[2] == self.resolution)
        return self.model.predict(_input)

    def process(self, _img):
        (_height, _width, _) = _img.shape
        inputs = []
        max_x = math.floor(_width / self.resolution)
        max_y = math.floor(_height / self.resolution)

        if _height == self.resolution and _width == self.resolution:
            inputs.append(_img)
        else:
            for x in range(0, max_x):
                for y in range(0, max_y):
                    inputs.append(_img[self.resolution * y:self.resolution * (y + 1),
                                  self.resolution * x:self.resolution * (x + 1)]
                                  )
        inputs = np.array(inputs, dtype='uint8')
        inputs = inputs / 255

        prediction = self.predict(inputs)
        print(prediction)
        assert len(prediction) == max_x * max_y
        _overlay = _img.copy()
        _output = _img.copy()
        if _CONVERT_TO_HSL:
            _output = cv2.cvtColor(_output, cv2.COLOR_HLS2RGB)
        else:
            _output = cv2.cvtColor(_output, cv2.COLOR_RGB2BGR)

        for x in range(0, max_x):
            for y in range(0, max_y):
                _index = (x*max_y) + y
                _color = self.true_color if prediction[_index] > 0.5 + self.threshold else \
                    self.false_color if prediction[_index] < 0.5 - self.threshold else (0,0,0)
                cv2.rectangle(_overlay,
                              (self.resolution * x, self.resolution * y),
                              (self.resolution * (x+1), self.resolution * (y + 1)),
                              _color,
                              -1)
                text = "YES" if prediction[_index] > 0.5 + self.threshold else "NO"
                cv2.putText(_output, text, (self.resolution * x + 30, self.resolution * y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        alpha = 0.1
        cv2.addWeighted(_overlay, alpha, _output, 1 - alpha,
                        0, _output)
        if _width > 1300:
            _output = cv2.resize(_output, (int(_width / 2), int(_height / 2)))
        cv2.imshow('Processed', _output)
        return _output


def __test__(filenames):
    # Self-testing, comment if not needed
    # data = cv2.imread('road2/road6_12.jpg')
    # data = cv2.imread('img/horverval2.jpg')
    for filename in filenames:
        data = cv2.imread(filename)
        (height, width, _) = data.shape

        detector = ConvDetector(_MODEL_FILENAME)

        if _CONVERT_TO_HSL:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2HLS)
        else:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        detector.process(data)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    __test__(['train/generated/empty/_0_5_road_x_17.png'
            , 'train/generated/empty/_17_0_road_x_5.png'
            , 'train/generated/road/_0_6_road_x_1.png'
            , 'train/generated/road/_0_6_road_x_5.png'
            ,'img/lanes9.jpg'
            , 'img/lanes2.jpg'
            ,'img/lanes8.jpg'
            ,'img/lanes6.jpg'
            ,'img/lanes1.jpg'
            ])

