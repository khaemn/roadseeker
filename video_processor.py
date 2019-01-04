from keras.models import Sequential, load_model
import numpy as np
import cv2

_MODEL_FILENAME = 'models/model_vae_roader.h5'

class RoadDetector:
    model = Sequential()
    max_RGB = 255

    input_height = 90
    input_width = 160
    mask_threshold = 200

    def __init__(self, modelFile=_MODEL_FILENAME):
        self.model = load_model(modelFile)

    def predict(self, _input):
        (original_height, original_width, _) = _input.shape
        # Resizing to acceptable size
        resized = cv2.resize(_input, (self.input_width, self.input_height))
        # Normalization
        normalized = resized / self.max_RGB
        # Prediction
        model_input = np.array([normalized])
        # Only one frame is used for prediction
        prediction = self.model.predict(model_input)
        prediction = prediction[0] * self.max_RGB

        prediction = prediction.astype('uint8')
        # prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2GRAY)
        processed = cv2.resize(prediction, (original_width, original_height), interpolation=cv2.INTER_LANCZOS4)
        _, mask = cv2.threshold(processed,
                                self.mask_threshold,
                                self.max_RGB,
                                cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)

        return mask

def simple_test():
    detector = RoadDetector()
    roadimg = cv2.imread('dataset/test/lanes1_r.jpg')
    roadimg = cv2.cvtColor(roadimg, cv2.COLOR_BGR2RGB)

    roadmask = detector.predict(roadimg)

    cv2.imshow('Origin', roadimg)
    cv2.imshow('Prediction', roadmask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(path):
    cam = cv2.VideoCapture(path)
    # Instal codecs using    $ sudo apt-get install ubuntu-restricted-extras
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    wr_width = 1024  # 640
    wr_height = 576  # 360
    # out = cv2.VideoWriter('output.avi', -1, 20.0, (wr_width, wr_height))
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (wr_width, wr_height))
    detector = RoadDetector()

    while True:
        ret_val, original = cam.read()
        if not ret_val:
            print("No video frame captured: video at end or no video present.")
            quit()

        original = cv2.resize(original, (wr_width, wr_height))

        dataForNN = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        prediction = detector.predict(dataForNN)

        alpha = 0.3
        combined = np.array(original, dtype=np.uint8)
        color_fill = np.array(original, dtype=np.uint8)
        color_fill[:, :] = [255, 50, 255]
        color_fill = cv2.bitwise_and(color_fill, color_fill, mask=prediction)
        cv2.addWeighted(combined, 1 - alpha, color_fill, alpha, 0, combined)

        cv2.imshow('Prediction', combined)

        out.write(combined)

        # cv2.imshow('Origin', cv2.cvtColor(dataForNN, cv2.COLOR_RGB2BGR))
        # cv2.imshow('Roadmask', prediction)

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cam.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_video('video/road6.mp4')