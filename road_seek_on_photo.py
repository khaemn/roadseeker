from roadseeker import *


def process_image(_image):

    while True:
        raw = imutils.resize(cv2.imread(_image), 640, 480)

        hsv = processor.toHls(raw)
        # cv2.imshow('HSV', hsl)

        # asphalt = processor.calcAsphaltColors3Hist(raw)
        # cv2.imshow('Asphalt HSL', asphalt)

        # thresholdedAnd = processor.thresholdedByAnd(hsl)
        # cv2.imshow('And Threshold HSL', thresholdedAnd)

        thresholdedGr = processor.thresholdedBy3Hist(hsv)
        # cv2.imshow('Or Threshold HSL', thresholdedOr)

        # thresholdedOnlyHue = processor.thresholdedOnlyHue(hsl)
        # cv2.imshow('Hue Threshold HSL', thresholdedOnlyHue)

        cutted = processor.plotInfo(raw)
        cv2.imshow('Processed HSL', np.concatenate((cutted, thresholdedGr), axis=1))



        # asphalt = processor.calcAsphaltColorsKmean(raw)
        # # cv2.imshow('Asphalt', asphalt)
        #
        # # thresholdedAnd = processor.thresholdedByAnd(img)
        # # cv2.imshow('And Threshold', thresholdedAnd)
        #
        # thresholdedGr = processor.thresholdedAfterUndersample(raw)
        # #cv2.imshow('Or Threshold', thresholdedOr)
        #
        # # thresholdedOnlyHue = processor.thresholdedOnlyHue(img)
        # # cv2.imshow('Hue Threshold', thresholdedOnlyHue)
        #
        # cutted = processor.plotInfo(raw)
        # #cv2.imshow('Cutted', cutted)
        # cv2.imshow('Processed RGB', np.concatenate((cutted, thresholdedGr), axis=1))

        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            quit()


processor = RoadSeeker()
processor.n_road_colors = 1
#processor.callback = process_image

process_image('img/lanes2.jpg')