from webcam import *


def process_image(_image):

    while True:
        img = imutils.resize(cv2.imread(_image), 640)

        #hsl = processor.toHls(img)
        #cv2.imshow('HSV', hsl)

        asphalt = processor.takeAsphaltRegion(img)
        cv2.imshow('Asphalt', asphalt)

        thresholdedAnd = processor.thresholdedByAnd(img)
        cv2.imshow('And Threshold', thresholdedAnd)

        thresholdedOnlyHue = processor.thresholdedOnlyHue(img)
        cv2.imshow('Hue Threshold', thresholdedOnlyHue)

        cutted = processor.plotInfo(img)
        edges = processor. edges(thresholdedAnd)

        #cv2.imshow('Edges', processor.edges(edges))
        cv2.imshow('Cutted', cutted)

        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            quit()


processor = RoadSeeker()
#processor.callback = process_image

process_image('road2/road2_1.jpg')