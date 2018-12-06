from roadseeker import *

from PIL import Image
import mss as MSS
import convroaddetector as CVD

monitor = {"top": 50, "left": 64, "width": 800, "height": 600}
output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)


def process_image():
    sct = MSS.mss()

    while True:
        sct_img = sct.grab(monitor)

        img = Image.frombytes('RGB', (sct_img.width, sct_img.height), sct_img.rgb)

        raw = np.array(img)

        hls = processor.toHls(raw)

        thresholded = processor.thresholdedBy3Hist(hls)

        cutted = processor.plotInfo(hls)
        cv2.imshow('Processed HSL', np.concatenate((cutted, thresholded), axis=1))

        if cv2.waitKey(10) == 27:
            cv2.destroyAllWindows()
            quit()

def process_monitor():
    sct = MSS.mss()
    detector = CVD.ConvDetector(CVD._MODEL_FILENAME)

    while True:
        sct_img = sct.grab(monitor)

        img = Image.frombytes('RGB', (sct_img.width, sct_img.height), sct_img.rgb)

        raw = np.array(img)

        detector.process(raw)

        if cv2.waitKey(10) == 27:
            cv2.destroyAllWindows()
            quit()

if __name__ == "__main__":
    processor = RoadSeeker()
    process_monitor()










# asphalt = processor.calcAsphaltColorsKmean(raw)
# # cv2.imshow('Asphalt', asphalt)
#
# # thresholdedAnd = processor.thresholdedByAnd(img)
# # cv2.imshow('And Threshold', thresholdedAnd)
#
# thresholdedOr = processor.thresholdedByOr(raw)
# # cv2.imshow('Or Threshold', thresholdedOr)
#
# # thresholdedOnlyHue = processor.thresholdedOnlyHue(img)
# # cv2.imshow('Hue Threshold', thresholdedOnlyHue)
#
# cutted = processor.plotInfo(raw)
# # cv2.imshow('Cutted', cutted)
# cv2.imshow('Processed RGB', np.concatenate((cutted, thresholdedOr), axis=1))
