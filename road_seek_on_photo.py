from webcam import *


def process_image(_image):

    while True:
        raw = imutils.resize(cv2.imread(_image), 640)

        hls = processor.toHls(raw)
        # cv2.imshow('HSV', hsl)

        asphalt = processor.calcAsphaltColorsHistHLS(hls)
        # cv2.imshow('Asphalt HSL', asphalt)

        # thresholdedAnd = processor.thresholdedByAnd(hsl)
        # cv2.imshow('And Threshold HSL', thresholdedAnd)

        thresholdedOr = processor.thresholdedOnlyHue(hls)
        # cv2.imshow('Or Threshold HSL', thresholdedOr)

        # thresholdedOnlyHue = processor.thresholdedOnlyHue(hsl)
        # cv2.imshow('Hue Threshold HSL', thresholdedOnlyHue)

        cutted = processor.plotInfo(hls)
        cv2.imshow('Processed HSL', np.concatenate((cutted, thresholdedOr), axis=1))



        asphalt = processor.calcAsphaltColorsKmean(raw)
        # cv2.imshow('Asphalt', asphalt)

        # thresholdedAnd = processor.thresholdedByAnd(img)
        # cv2.imshow('And Threshold', thresholdedAnd)

        thresholdedOr = processor.thresholdedByOr(raw)
        #cv2.imshow('Or Threshold', thresholdedOr)

        # thresholdedOnlyHue = processor.thresholdedOnlyHue(img)
        # cv2.imshow('Hue Threshold', thresholdedOnlyHue)

        cutted = processor.plotInfo(raw)
        #cv2.imshow('Cutted', cutted)
        cv2.imshow('Processed RGB', np.concatenate((cutted, thresholdedOr), axis=1))

        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            quit()


processor = RoadSeeker()
processor.n_road_colors = 1
#processor.callback = process_image

process_image('img/lanes3.png')