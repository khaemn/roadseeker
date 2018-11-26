from webcam import *

from PIL import Image
import mss as MSS

monitor = {"top": 500, "left": 500, "width": 500, "height": 500}
output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)


def process_image():
    sct = MSS.mss()

    while True:
        sct_img = sct.grab(monitor)

        img = Image.frombytes('RGB', (sct_img.width, sct_img.height), sct_img.rgb)

        img = np.array(img)

        #hsl = processor.toHls(img)
        #cv2.imshow('HSV', hsl)

        asphalt = processor.takeAsphaltRegion(img)
        #cv2.imshow('Asphalt', asphalt)

        thresholdedAnd = processor.thresholdedByAnd(img)
        cv2.imshow('And Threshold', thresholdedAnd)

        #thresholdedOnlyHue = processor.thresholdedOnlyHue(img)
        #cv2.imshow('Hue Threshold', thresholdedOnlyHue)

        cutted = processor.plotInfo(img)
        #edges = processor. edges(thresholdedAnd)

        #cv2.imshow('Edges', processor.edges(edges))
        cv2.imshow('Cutted', cutted)

        if cv2.waitKey(10) == 27:
            cv2.destroyAllWindows()
            quit()


processor = RoadSeeker()

process_image()