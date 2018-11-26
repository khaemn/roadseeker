import cv2
import imutils
from collections import namedtuple
import typing
import numpy as np
import edge

Point = typing.NamedTuple("Point", [('x', int), ('y', int)])

# Areas should be given using 0..1 range, where 1 is input width or height.
SearchArea = namedtuple('SearchArea', 'left, top, width, height')


_SETTINGS_W = 'Settings'
_ASPHALT_LEFT = 'Box left'
_MIN_LIGHTNESS = 'Lightness min'
_MIN_RED = 'Lightness min'
_ERODE_KERNEL = 'Erode kernel'
_DILATE_KERNEL = 'Dilate kernel'


class RoadSeeker:
    
    top_cutting_ratio = 0.4  # defines the cut top rectangle of the frame
    road_search_area = SearchArea(.45, .85, .1, .1)
    insignificant_area = SearchArea(0., 0., 1., .4)
    horizon = 0.45  # horizon line in respectance to frame heightm starting from top
    line_width = 2
    search_color = (0, 255, 0)
    asphalt_colors = [(0, 0, 0)]
    asphalt_spread = 10  # in RGB units, defines the thresholding corridor
    min_lightness = 120
    min_red = 110
    canny_min = 100
    canny_max = 200
    erode_kernel = 2
    dilate_kernel = 2

    n_road_colors = 10  # top N most relevant colors in the asphalt rectangle will be taken into account

    def __init__(self):
        cv2.namedWindow(_SETTINGS_W)
        # create trackbars for color change
        cv2.createTrackbar('Threshold spread', _SETTINGS_W, 1, 100, RoadSeeker.nothing)
        cv2.setTrackbarPos('Threshold spread', _SETTINGS_W, self.asphalt_spread)

        cv2.createTrackbar('Erode kernel', _SETTINGS_W, 1, 30, RoadSeeker.nothing)
        cv2.setTrackbarPos('Erode kernel', _SETTINGS_W, self.erode_kernel)

        cv2.createTrackbar('Dilate kernel', _SETTINGS_W, 1, 30, RoadSeeker.nothing)
        cv2.setTrackbarPos('Dilate kernel', _SETTINGS_W, self.dilate_kernel)

        cv2.createTrackbar('Canny min', _SETTINGS_W, 1, 200, RoadSeeker.nothing)
        cv2.setTrackbarPos('Canny min', _SETTINGS_W, self.canny_min)
        cv2.createTrackbar('Canny max', _SETTINGS_W, 1, 200, RoadSeeker.nothing)
        cv2.setTrackbarPos('Canny max', _SETTINGS_W, self.canny_max)

        cv2.createTrackbar('Box left', _SETTINGS_W, 1, 100, RoadSeeker.nothing)
        cv2.setTrackbarPos('Box left', _SETTINGS_W, int(self.road_search_area.left * 100))
        cv2.createTrackbar('Box top', _SETTINGS_W, 1, 100, RoadSeeker.nothing)
        cv2.setTrackbarPos('Box top', _SETTINGS_W, int(self.road_search_area.top * 100))

        cv2.createTrackbar('Lightness min', _SETTINGS_W, 1, 255, RoadSeeker.nothing)
        cv2.setTrackbarPos('Lightness min', _SETTINGS_W, self.min_lightness)

        cv2.createTrackbar('Lightness min', _SETTINGS_W, 1, 255, RoadSeeker.nothing)
        cv2.setTrackbarPos('Lightness min', _SETTINGS_W, self.min_lightness)

        cv2.createTrackbar('R min', _SETTINGS_W, 1, 255, RoadSeeker.nothing)
        cv2.setTrackbarPos('R min', _SETTINGS_W, self.min_red)

        cv2.createTrackbar('Asphalt colors', _SETTINGS_W, 1, 20, RoadSeeker.nothing)
        cv2.setTrackbarPos('Asphalt colors', _SETTINGS_W, self.n_road_colors)

    @staticmethod
    def nothing(_):
        pass

    @staticmethod  # takes w and h in pixels, area of float coeffs, returns top-left and bottom-right vertexes
    def rectangleVertexes(width, height, area):
        top_left = Point(int(width * area.left),
                         int(height * area.top))
        bottom_right = Point(int(width * (area.left + area.width)),
                             int(height * (area.top + area.height)))
        return top_left, bottom_right


    def shrink(self, _input):
        _width, _height = _input.shape[1], _input.shape[0]

        #  cut the topmost
        #  result = cv2.resize(_input, (400, 200))
        processed = _input

        # Show road searching area
        p1, p2 = RoadSeeker.rectangleVertexes(_width, _height, self.road_search_area)
        cv2.rectangle(processed,
                      (int(p1.x), int(p1.y)),
                      (int(p2.x), int(p2.y)),
                      self.search_color,
                      self.line_width)

        self.plotAsphaltColors(processed)

        return processed

    def plotAsphaltColors(self, _input):
        _width, _height = _input.shape[1], _input.shape[0]
        # draw insignificant area using top 3 colors in asphalt road
        rectangle_width = int(_width / self.n_road_colors)
        for i in range(0, len(self.asphalt_colors)):
            cv2.rectangle(_input,
                          (rectangle_width * i, 0),
                          (rectangle_width * (i+1), int(_height * self.horizon)),
                          self.asphalt_colors[i],
                          -1)

    def takeAsphaltRegion(self, _input):
        _k_mean_shrink_ratio = 2
        _width, _height = _input.shape[1], _input.shape[0]

        self.road_search_area = SearchArea(cv2.getTrackbarPos('Box left', _SETTINGS_W) / 100,
                                           cv2.getTrackbarPos('Box top', _SETTINGS_W) / 100,
                                           self.road_search_area.width,
                                           self.road_search_area.height)

        p1, p2 = RoadSeeker.rectangleVertexes(_width, _height, self.road_search_area)
        asphalt = _input[p1.y:p2.y, p1.x:p2.x]

        undersampled = cv2.resize(asphalt,
                                  (int(_width / _k_mean_shrink_ratio),
                                   int(_height / _k_mean_shrink_ratio)))
        asphalt = cv2.resize(undersampled, (_width, _height))

        # Extracting top N relevant asphalt colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    200,  # 200,
                    .5)  # .1)
        pixels = np.float32(undersampled.reshape(-1, 3))
        flags = cv2.KMEANS_RANDOM_CENTERS

        self.n_road_colors = cv2.getTrackbarPos('Asphalt colors', _SETTINGS_W) + 1
        _, labels, palette = cv2.kmeans(pixels, self.n_road_colors, None, criteria, 1, flags)

        self.asphalt_colors.clear()
        while len(self.asphalt_colors) < self.n_road_colors:
            self.asphalt_colors.append((0, 0, 0))

        for i in range(0, min(self.n_road_colors, len(palette))):
            dominant = palette[i]
            self.asphalt_colors[i] = (int(dominant[0]),
                                      int(dominant[1]),
                                      int(dominant[2]))

        return asphalt

    def toHls(self, _input):
        hls = cv2.cvtColor(_input, cv2.COLOR_BGR2HLS)
        return hls

    def thresholdedRoad(self, _input):
        self.asphalt_spread = cv2.getTrackbarPos('Threshold spread', _SETTINGS_W) + 1
        self.erode_kernel = cv2.getTrackbarPos('Erode kernel', _SETTINGS_W) + 1
        self.dilate_kernel = cv2.getTrackbarPos('Dilate kernel', _SETTINGS_W) + 1
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.erode_kernel, self.erode_kernel * 2))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate_kernel, self.dilate_kernel * 2))

        fullmask = cv2.inRange(_input, (0,0,0), (255,255,255))
        for i in range (0, self.n_road_colors):
            lower_color_bounds = np.array([bound * (self.asphalt_spread / 100) for bound in self.asphalt_colors[i]])
            upper_color_bounds = np.array([bound / (self.asphalt_spread / 100) for bound in self.asphalt_colors[i]])

            mask = cv2.inRange(_input, lower_color_bounds, upper_color_bounds)

            mask = cv2.erode(mask, erode_kernel)
            mask = cv2.dilate(mask, dilate_kernel)

            fullmask = mask if i == 0 else cv2.bitwise_and(fullmask, mask)
            # res = cv2.bitwise_and(_input, _input, mask=mask)
            # cv2.imshow('Mask ' + str(i), mask)
        cv2.imshow('Fullmask ', fullmask)
        mask_rgb = cv2.cvtColor(fullmask, cv2.COLOR_GRAY2BGR)
        frame = _input & mask_rgb

        return frame

    def edges(self, _input):
        self.canny_min = cv2.getTrackbarPos('Canny min', _SETTINGS_W)
        self.canny_min = cv2.getTrackbarPos('Canny max', _SETTINGS_W)
        edges = cv2.Canny(_input, self.canny_min, self.canny_min)
        return edges

    def significant(self, _input):
        _width, _height = _input.shape[1], _input.shape[0]
        return _input[int(_height * self.horizon):_height, 0:_width]

    def erode(self, _input):
        _width, _height = _input.shape[1], _input.shape[0]
        eroded = cv2.erode(_input)
        return _input[int(_height * self.horizon):_height, 0:_width]

    def driving_lane(self, image):
        # function that finds the road driving lane line
        # image : camera image where the line locations are to be located
        # return : a masked image of onlt the lane lines
        self.min_lightness = cv2.getTrackbarPos('Lightness min', _SETTINGS_W)
        self.min_red = cv2.getTrackbarPos('R min', _SETTINGS_W)
        # Convert to HSV color space and separate the V channel
        # hls for Sobel edge detection
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        # use on the luminance channel data for edges
        # TODO: implement edge.binary_array in pycuda for speed
        _, sxbinary = edge.threshold(hls[:, :, 1], thresh=(self.min_lightness, 255))
        sxbinary = edge.blur_gaussian(sxbinary, ksize=3)

        # find the edges in the channel data using sobel magnitudes
        sxbinary = edge.mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))

        s_channel = hls[:, :, 2]  # use only the saturation channel data
        _, s_binary = edge.threshold(s_channel, (self.min_lightness, 255))
        _, r_thresh = edge.threshold(image[:, :, 2], thresh=(self.min_red, 255))

        rs_binary = cv2.bitwise_and(s_binary, r_thresh)
        return cv2.bitwise_or(rs_binary, sxbinary.astype(np.uint8))
    
    def roi(self, _input):
        


def process_video(input=0, mirror=False):
    processor = RoadSeeker()
    cam = cv2.VideoCapture(input)

    while True:
        ret_val, raw = cam.read()
        if not ret_val:
            print("No video frame captured: video at end or no video present.")
            quit()
        raw = imutils.resize(raw, int(raw.shape[1]/2))
        # if mirror:
        #   img = cv2.flip(img, 1)

        # cv2.imshow('Input', img)

        #hsv = processor.toHls(raw)
        #cv2.imshow('HSV', hsv)

        asphalt = processor.takeAsphaltRegion(raw)
        # cv2.imshow('Asphalt', asphalt)

        significant = processor.significant(raw)

        thresholded = processor.thresholdedRoad(significant)
        cv2.imshow('Threshold', thresholded)

        cutted = processor.shrink(raw)
        edges = processor.edges(thresholded)

        cv2.imshow('Edges', processor.edges(edges))
        cv2.imshow('Cutted', cutted)

        cv2.imshow('Lane', processor.driving_lane(raw))


        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()

process_video('road2.mp4')
# process_video()