from keras.models import Sequential, load_model

_MODEL_FILENAME = 'models/model_road_detector.h5'


class ConvDetector:
    model = Sequential()
    resolution = 100  # pixels

    def __init__(self, modelFile):
        self.model = load_model(modelFile)

    def predict(self, input):
        assert(input.ndim == 4
               and input.shape[1] == self.resolution
               and input.shape[2] == self.resolution)
        return self.model.predict(input)



# Self-testing, comment if not needed
from PIL import Image
import numpy as np

img = Image.open('road2/road6_0.jpg')
data = np.asarray(img, dtype='uint8')

detector = ConvDetector(_MODEL_FILENAME)
inputs = []
for x in range(0, 10):
    for y in range(0, 7):
        inputs.append(data[100*y:100*(y+1), 100*x:100*(x+1)])
inputs = np.array(inputs)

prediction = detector.predict(inputs)

print(str(len(prediction)), "predictions made:\n", prediction)



