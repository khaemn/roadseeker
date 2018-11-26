from webcam import *

from PIL import Image
import mss as MSS

monitor = {"top": 20, "left": 64, "width": 800, "height": 600}
output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)


def process_image():
    sct = MSS.mss()

    while True:
        sct_img = sct.grab(monitor)

        img = Image.frombytes('RGB', (sct_img.width, sct_img.height), sct_img.rgb)

        raw = np.array(img)

        hls = processor.toHls(raw)
        # cv2.imshow('HSV', hsl)

        # asphalt = processor.calcAsphaltColorsHistHLS(hls)
        # cv2.imshow('Asphalt HSL', asphalt)

        # thresholdedAnd = processor.thresholdedByAnd(hsl)
        # cv2.imshow('And Threshold HSL', thresholdedAnd)

        thresholdedOr = processor.thresholdedBy3Hist(hls)
        # cv2.imshow('Or Threshold HSL', thresholdedOr)

        # thresholdedOnlyHue = processor.thresholdedOnlyHue(hsl)
        # cv2.imshow('Hue Threshold HSL', thresholdedOnlyHue)

        cutted = processor.plotInfo(hls)
        cv2.imshow('Processed HSL', np.concatenate((cutted, thresholdedOr), axis=1))

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

        if cv2.waitKey(10) == 27:
            cv2.destroyAllWindows()
            quit()


processor = RoadSeeker()

process_image()