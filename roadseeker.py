# Inspired by Udacity https://github.com/Heych88/udacity-sdcnd-advanced-lane-finding/

import cv2
import imutils
from collections import namedtuple
import typing
import numpy as np
import edge
import heapq
import convroaddetector as CVD

Point = typing.NamedTuple("Point", [('x', int), ('y', int)])

# Areas should be given using 0..1 range, where 1 is input width or height.
SearchArea = namedtuple('SearchArea', 'left, top, width, height')


str_SETTINGS_W = 'Settings'

str_ASPHALT_LEFT = 'Asphalt box left'
str_ASPHALT_TOP = 'Asphalt box top'
str_ASPHALT_WIDTH = 'Asphalt box width'
str_ASPHALT_COLORS = 'N asphalt colors'

str_MIN_LIGHTNESS = 'Lightness min'
str_MIN_RED = 'R min'
str_FOCUS_X= 'Focus X'
str_FOCUS_Y= 'Focus Y'
str_ERODE_KERNEL = 'Erode kernel'
str_DILATE_KERNEL = 'Dilate kernel'
str_ASPHALT_HUE_THRESHOLD_SPREAD = 'Hue spread'
str_ASPHALT_SAT_THRESHOLD_SPREAD = 'Sat spread'
str_ASPHALT_LIG_THRESHOLD_SPREAD = 'Lig spread'
str_MIN_CANNY = 'Canny min'
str_MAX_CANNY = 'Canny max'

_MAX_RGB = 255
_VIDEO_WIDTH = 640
_HUNDRED = 100

class RoadSeeker:

    road_search_area = SearchArea(.45, .85, .1, .1)
    insignificant_area = SearchArea(0., 0., 1., .4)
    horizon = 0.35  # horizon line in respectance to frame heightm starting from top
    focal_point = Point(0, 0)
    roi_region = [Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0)]
    line_width = 2
    search_color = (0, 255, 0)
    asphalt_colors = [(0, 0, 0)]

    asphalt_hue_spread = 10  # in RGB units, defines the thresholding corridor
    asphalt_sat_spread = 10  # in RGB units, defines the thresholding corridor
    asphalt_lig_spread = 10  # in RGB units, defines the thresholding corridor


    min_lightness = 120
    min_red = 4
    canny_min = 100
    canny_max = 200
    erode_size = 2
    dilate_size = 2
    erode_kernel = 0
    dilate_kernel = 0

    n_road_colors = 3  # top N most relevant colors in the asphalt rectangle will be taken into account


    def __init__(self):
        cv2.namedWindow(str_SETTINGS_W)

        # create settings trackbars
        cv2.createTrackbar(str_ASPHALT_HUE_THRESHOLD_SPREAD, str_SETTINGS_W, 1, _HUNDRED, self.control_moved)
        cv2.setTrackbarPos(str_ASPHALT_HUE_THRESHOLD_SPREAD, str_SETTINGS_W, self.asphalt_hue_spread)

        cv2.createTrackbar(str_ASPHALT_SAT_THRESHOLD_SPREAD, str_SETTINGS_W, 1, _HUNDRED, self.control_moved)
        cv2.setTrackbarPos(str_ASPHALT_SAT_THRESHOLD_SPREAD, str_SETTINGS_W, self.asphalt_sat_spread)

        cv2.createTrackbar(str_ASPHALT_LIG_THRESHOLD_SPREAD, str_SETTINGS_W, 1, _HUNDRED, self.control_moved)
        cv2.setTrackbarPos(str_ASPHALT_LIG_THRESHOLD_SPREAD, str_SETTINGS_W, self.asphalt_lig_spread)

        cv2.createTrackbar(str_ERODE_KERNEL, str_SETTINGS_W, 1, 30, self.control_moved)
        cv2.setTrackbarPos(str_ERODE_KERNEL, str_SETTINGS_W, self.erode_size)

        cv2.createTrackbar(str_DILATE_KERNEL, str_SETTINGS_W, 1, 30, self.control_moved)
        cv2.setTrackbarPos(str_DILATE_KERNEL, str_SETTINGS_W, self.dilate_size)

        # cv2.createTrackbar(str_MIN_CANNY, str_SETTINGS_W, 1, 200, self.control_moved)
        # cv2.setTrackbarPos(str_MIN_CANNY, str_SETTINGS_W, self.canny_min)
        # cv2.createTrackbar(str_MAX_CANNY, str_SETTINGS_W, 1, 200, self.control_moved)
        # cv2.setTrackbarPos(str_MAX_CANNY, str_SETTINGS_W, self.canny_max)

        cv2.createTrackbar(str_ASPHALT_LEFT, str_SETTINGS_W, 1, 100, self.control_moved)
        cv2.setTrackbarPos(str_ASPHALT_LEFT, str_SETTINGS_W, int(self.road_search_area.left * 100))
        cv2.createTrackbar(str_ASPHALT_TOP, str_SETTINGS_W, 1, 100, self.control_moved)
        cv2.setTrackbarPos(str_ASPHALT_TOP, str_SETTINGS_W, int(self.road_search_area.top * 100))
        cv2.createTrackbar(str_ASPHALT_WIDTH, str_SETTINGS_W, 1, 100, self.control_moved)
        cv2.setTrackbarPos(str_ASPHALT_WIDTH, str_SETTINGS_W, int(self.road_search_area.width * 100))

        cv2.createTrackbar(str_FOCUS_X, str_SETTINGS_W, 1, 100, self.control_moved)
        cv2.setTrackbarPos(str_FOCUS_X, str_SETTINGS_W, 50)
        cv2.createTrackbar(str_FOCUS_Y, str_SETTINGS_W, 1, 100, self.control_moved)
        cv2.setTrackbarPos(str_FOCUS_Y, str_SETTINGS_W, 50)

        cv2.createTrackbar(str_MIN_LIGHTNESS, str_SETTINGS_W, 1, 255, self.control_moved)
        cv2.setTrackbarPos(str_MIN_LIGHTNESS, str_SETTINGS_W, self.min_lightness)

        cv2.createTrackbar(str_MIN_LIGHTNESS, str_SETTINGS_W, 1, 255, self.control_moved)
        cv2.setTrackbarPos(str_MIN_LIGHTNESS, str_SETTINGS_W, self.min_lightness)

        cv2.createTrackbar(str_MIN_RED, str_SETTINGS_W, 1, 255, self.control_moved)
        cv2.setTrackbarPos(str_MIN_RED, str_SETTINGS_W, self.min_red)

        cv2.createTrackbar(str_ASPHALT_COLORS, str_SETTINGS_W, 1, 50, self.control_moved)
        cv2.setTrackbarPos(str_ASPHALT_COLORS, str_SETTINGS_W, self.n_road_colors)

    def control_moved(self, _):
        return

    @staticmethod  # takes w and h in pixels, area of float coeffs, returns top-left and bottom-right vertexes
    def rectangleVertexes(width, height, area):
        top_left = Point(int(width * area.left),
                         int(height * area.top))
        bottom_right = Point(int(width * (area.left + area.width)),
                             int(height * (area.top + area.height)))
        return top_left, bottom_right


    def plotInfo(self, _input):
        _width, _height = _input.shape[1], _input.shape[0]

        # Show road searching area
        p1, p2 = RoadSeeker.rectangleVertexes(_width, _height, self.road_search_area)
        cv2.rectangle(_input,
                      (int(p1.x), int(p1.y)),
                      (int(p2.x), int(p2.y)),
                      self.search_color,
                      self.line_width)

        self.plotAsphaltColors(_input)

        return _input

    def readAsphaltBox(self, _input):
        (_height, _width, _) = _input.shape
        self.n_road_colors = cv2.getTrackbarPos(str_ASPHALT_COLORS, str_SETTINGS_W) + 1
        self.road_search_area = SearchArea(cv2.getTrackbarPos(str_ASPHALT_LEFT, str_SETTINGS_W) / 100,
                                           cv2.getTrackbarPos(str_ASPHALT_TOP, str_SETTINGS_W) / 100,
                                           cv2.getTrackbarPos(str_ASPHALT_WIDTH, str_SETTINGS_W) / 100,
                                           self.road_search_area.height)
        focus_x = int(_width * cv2.getTrackbarPos(str_FOCUS_X, str_SETTINGS_W) / 100)
        focus_y = int(_width * cv2.getTrackbarPos(str_FOCUS_Y, str_SETTINGS_W) / 100)
        self.focal_point = Point(focus_x, focus_y)
        p1, p2 = RoadSeeker.rectangleVertexes(_width, _height, self.road_search_area)
        asphalt = _input[p1.y:p2.y, p1.x:p2.x]
        return asphalt

    def plotAsphaltColors(self, _input):
        (_height, _width, _) = _input.shape
        # draw insignificant area using top N colors in asphalt road
        rectangle_width = int(_width / self.n_road_colors)
        for i in range(0, len(self.asphalt_colors)):
            cv2.rectangle(_input,
                          (rectangle_width * i, 0),
                          (rectangle_width * (i+1), int(_height * self.horizon)),
                          self.asphalt_colors[i],
                          -1)

    def calcAsphaltColorsKmean(self, _input):
        _k_mean_shrink_ratio = 10
        (_height, _width, _) = _input.shape

        self.readAsphaltBox(_input)

        undersampled = imutils.resize(self.readAsphaltBox(_input), (int(_width / _k_mean_shrink_ratio)))

        # Extracting top N relevant asphalt colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    200,  # 200,
                    .5)  # .1)
        pixels = np.float32(undersampled.reshape(-1, 3))
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, labels, palette = cv2.kmeans(pixels, self.n_road_colors, None, criteria, 1, flags)
        _, counts = np.unique(labels, return_counts=True)

        # Have tried to use no kmean, but average - works not very fine.
        # palette = [_input.mean(axis=0).mean(axis=0)]
        # print(palette[0])

        self.asphalt_colors.clear()
        while len(self.asphalt_colors) < self.n_road_colors:
            self.asphalt_colors.append((0, 0, 0))

        for i in range(0, min(self.n_road_colors, len(palette))):
            dominant = palette[i]
            self.asphalt_colors[i] = (int(dominant[0]),
                                      int(dominant[1]),
                                      int(dominant[2]))

        return cv2.resize(undersampled, (_width, _height))

    def calcAsphaltColorsHistHLS(self, _input):
        _hist_shrink_ratio = 1
        (_height, _width, _) = _input.shape

        self.readAsphaltBox(_input)

        undersampled = imutils.resize(self.readAsphaltBox(_input), (int(_width / _hist_shrink_ratio)))

        hist = cv2.calcHist([undersampled],  # images
                            [0],             # channels
                            None,            # mask
                            [_MAX_RGB],      # histSize
                            [0, _MAX_RGB])   # ranges

        dcols = heapq.nlargest(self.n_road_colors, range(len(hist)), hist.take)

        dominating_color_hue = hist.argmax()

        # numpy approach to histing s nor the best
        mean_color = undersampled.mean(axis=0).mean(axis=0)

        self.asphalt_colors.clear()
        while len(self.asphalt_colors) < self.n_road_colors:
            self.asphalt_colors.append((0, 0, 0))

        for i in range(0, self.n_road_colors):
            dominant = [dcols[i], mean_color[1], mean_color[2]]
            self.asphalt_colors[i] = (int(dominant[0]),
                                      int(dominant[1]),
                                      int(dominant[2]))

        return cv2.resize(undersampled, (_width, _height))

    def calcAsphaltColorBoundaries(self, _input):
        _hist_shrink_ratio = 5
        (_height, _width, _) = _input.shape

        self.readAsphaltBox(_input)

        undersampled = imutils.resize(self.readAsphaltBox(_input), (int(_width / _hist_shrink_ratio)))

        hist0 = cv2.calcHist([undersampled],  # images
                             [0],             # channels
                             None,            # mask
                             [_MAX_RGB],      # histSize
                             [0, _MAX_RGB])   # ranges

        hist1 = cv2.calcHist([undersampled],  # images
                             [1],             # channels
                             None,            # mask
                             [_MAX_RGB],      # histSize
                             [0, _MAX_RGB])   # ranges

        hist2 = cv2.calcHist([undersampled],  # images
                             [2],             # channels
                             None,            # mask
                             [_MAX_RGB],      # histSize
                             [0, _MAX_RGB])   # ranges

        dcols0 = heapq.nlargest(self.n_road_colors, range(len(hist0)), hist0.take)
        dcols1 = heapq.nlargest(self.n_road_colors, range(len(hist1)), hist1.take)
        dcols2 = heapq.nlargest(self.n_road_colors, range(len(hist2)), hist2.take)

        lower_color_bounds = [min(dcols0), min(dcols1), min(dcols2)]
        upper_color_bounds = [max(dcols0), max(dcols1), max(dcols2)]

        self.asphalt_colors = [lower_color_bounds, upper_color_bounds]
        while len(self.asphalt_colors) < self.n_road_colors:
            self.asphalt_colors.append((0,0,0))

        return lower_color_bounds, upper_color_bounds


    def toHls(self, _input):
        hls = cv2.cvtColor(_input, cv2.COLOR_BGR2HLS)
        return hls

    def toHsv(self, _input):
        hsv = cv2.cvtColor(_input, cv2.COLOR_BGR2HSV)
        return hsv

    def thresholdedByAnd(self, _input):
        self.readThresholdingParams()

        fullmask = cv2.inRange(_input, (0,0,0), (255,255,255))  # full true map
        for i in range (0, self.n_road_colors):
            lower_color_bounds = np.array([bound - self.asphalt_hue_spread for bound in self.asphalt_colors[i]])
            upper_color_bounds = np.array([bound + self.asphalt_hue_spread for bound in self.asphalt_colors[i]])

            mask = cv2.inRange(_input, lower_color_bounds, upper_color_bounds)

            if self.erode_size > 0:
                mask = cv2.erode(mask, self.erode_kernel)
            if self.dilate_size > 0:
                mask = cv2.dilate(mask, self.dilate_kernel)

            fullmask = mask if i == 0 else cv2.bitwise_and(fullmask, mask)
        # cv2.imshow('Fullmask', fullmask)
        mask_rgb = cv2.cvtColor(fullmask, cv2.COLOR_GRAY2BGR)
        frame = _input & mask_rgb

        return frame

    def thresholdedByOr(self, _input):
        self.readThresholdingParams()

        fullmask = cv2.inRange(_input, (255,255,255), (0,0,0)) # full false map
        for i in range (0, self.n_road_colors):
            lower_color_bounds = np.array([bound - self.asphalt_hue_spread for bound in self.asphalt_colors[i]])
            upper_color_bounds = np.array([bound + self.asphalt_hue_spread for bound in self.asphalt_colors[i]])

            mask = cv2.inRange(_input, lower_color_bounds, upper_color_bounds)

            if self.erode_size > 0:
                mask = cv2.erode(mask, self.erode_kernel)
            if self.dilate_size > 0:
                mask = cv2.dilate(mask, self.dilate_kernel)

            fullmask = cv2.bitwise_or(fullmask, mask)
        # cv2.imshow('Fullmask', fullmask)
        mask_rgb = cv2.cvtColor(fullmask, cv2.COLOR_GRAY2BGR)
        frame = _input & mask_rgb

        return frame

    def thresholdedOnlyHue(self, _input):
        self.readThresholdingParams()

        fullmask = cv2.inRange(_input, (0, 0,  0), (_MAX_RGB,_MAX_RGB,_MAX_RGB)) # full true map
        for i in range(0, self.n_road_colors):
            lower_color_bounds = np.array([self.asphalt_colors[i][0] * ( 1 - self.asphalt_hue_spread),
                                           self.asphalt_colors[i][2] * ( 1 - self.asphalt_lig_spread),
                                           self.asphalt_colors[i][1] * ( 1 - self.asphalt_sat_spread)])
            upper_color_bounds = np.array([self.asphalt_colors[i][0] * ( 1 + self.asphalt_hue_spread),
                                           self.asphalt_colors[i][2] * ( 1 + self.asphalt_lig_spread),
                                           self.asphalt_colors[i][1] * ( 1 + self.asphalt_sat_spread)])

            mask = cv2.inRange(_input, lower_color_bounds, upper_color_bounds)

            if self.erode_size > 0:
                mask = cv2.erode(mask, self.erode_kernel)
            if self.dilate_size > 0:
                mask = cv2.dilate(mask, self.dilate_kernel)

            fullmask = cv2.bitwise_and(fullmask, mask)
        # cv2.imshow('Fullmask', fullmask)
        mask_rgb = cv2.cvtColor(fullmask, cv2.COLOR_GRAY2BGR)
        frame = _input & mask_rgb

        return frame

    def thresholdedGray(self, _input):
        self.readThresholdingParams()

        fullmask = cv2.inRange(_input, (_MAX_RGB,_MAX_RGB,_MAX_RGB), (0, 0,  0) ) # full true map
        for i in range(0, self.n_road_colors):
            lower_color_bounds = np.array([self.asphalt_colors[i][0] * ( 1 - self.asphalt_hue_spread),
                                           self.asphalt_colors[i][2] * ( 1 - self.asphalt_lig_spread),
                                           self.asphalt_colors[i][1] * ( 1 - self.asphalt_sat_spread)])
            upper_color_bounds = np.array([self.asphalt_colors[i][0] * ( 1 + self.asphalt_hue_spread),
                                           self.asphalt_colors[i][2] * ( 1 + self.asphalt_lig_spread),
                                           self.asphalt_colors[i][1] * ( 1 + self.asphalt_sat_spread)])

            mask = cv2.inRange(_input, lower_color_bounds, upper_color_bounds)
            if self.erode_size > 0:
                mask = cv2.erode(mask, self.erode_kernel)
            if self.dilate_size > 0:
                mask = cv2.dilate(mask, self.dilate_kernel)

            fullmask = cv2.addWeighted(fullmask, 0.1 * (1 + i), mask, 0.1, 0.0)
        cv2.imshow('Fullmask', fullmask)
        mask_rgb = cv2.cvtColor(fullmask, cv2.COLOR_GRAY2BGR)
        frame = _input & mask_rgb

        return frame

    def thresholdedBy3Hist(self, _input):
        self.readThresholdingParams()
        (_height, _width, _) = _input.shape
        resample_ratio = 5
        resamle = cv2.resize(_input, (int(_width/resample_ratio), int(_height/resample_ratio)))

        lo, hi = self.calcAsphaltColorBoundaries(_input)
        lo, hi = np.array(lo), np.array(hi)

        lo -= self.asphalt_hue_spread
        hi += self.asphalt_hue_spread

        fullmask = cv2.inRange(resamle, lo, hi ) # full true map
        if self.erode_size > 0:
            fullmask = cv2.erode(fullmask, self.erode_kernel)
        if self.dilate_size > 0:
            fullmask = cv2.dilate(fullmask, self.dilate_kernel)
        cv2.imshow('Fullmask', fullmask)
        mask_rgb = cv2.cvtColor(fullmask, cv2.COLOR_GRAY2BGR)
        mask_rgb = cv2.resize(mask_rgb, (_width, _height))
        mask_rgb = cv2.blur(mask_rgb, (self.erode_size+1, self.erode_size+1))
        frame = _input & mask_rgb

        return frame

    def thresholdedAfterUndersample(self, _input):
        self.readThresholdingParams()
        (_height, _width, _) = _input.shape
        _undersampling_ratio = 5
        unders = cv2.resize(_input, (int(_width / _undersampling_ratio), int(_height / _undersampling_ratio)))
        fullmask = cv2.inRange(unders, (_MAX_RGB,_MAX_RGB,_MAX_RGB), (0, 0,  0) ) # full true map
        for i in range(0, self.n_road_colors):
            lower_color_bounds = np.array([self.asphalt_colors[i][0] * ( 1 - self.asphalt_hue_spread),
                                           self.asphalt_colors[i][2] * ( 1 - self.asphalt_lig_spread),
                                           self.asphalt_colors[i][1] * ( 1 - self.asphalt_sat_spread)])
            upper_color_bounds = np.array([self.asphalt_colors[i][0] * ( 1 + self.asphalt_hue_spread),
                                           self.asphalt_colors[i][2] * ( 1 + self.asphalt_lig_spread),
                                           self.asphalt_colors[i][1] * ( 1 + self.asphalt_sat_spread)])

            mask = cv2.inRange(unders, lower_color_bounds, upper_color_bounds)
            if self.erode_size > 0:
                mask = cv2.erode(mask, self.erode_kernel)
            if self.dilate_size > 0:
                mask = cv2.dilate(mask, self.dilate_kernel)

            fullmask = cv2.bitwise_or(fullmask, mask)
        cv2.imshow('Fullmask', fullmask)
        mask_rgb = cv2.cvtColor(fullmask, cv2.COLOR_GRAY2BGR)
        mask_rgb = cv2.resize(mask_rgb, (_width, _height))
        frame = _input & mask_rgb

        return frame

    def readThresholdingParams(self):
        self.asphalt_hue_spread = cv2.getTrackbarPos(str_ASPHALT_HUE_THRESHOLD_SPREAD, str_SETTINGS_W) + 1
        self.asphalt_sat_spread = cv2.getTrackbarPos(str_ASPHALT_SAT_THRESHOLD_SPREAD, str_SETTINGS_W) + 1
        self.asphalt_lig_spread = cv2.getTrackbarPos(str_ASPHALT_LIG_THRESHOLD_SPREAD, str_SETTINGS_W) + 1
        self.erode_size = cv2.getTrackbarPos(str_ERODE_KERNEL, str_SETTINGS_W)
        self.dilate_size = cv2.getTrackbarPos(str_DILATE_KERNEL, str_SETTINGS_W)
        if self.erode_size > 0:
            self.erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.erode_size, self.erode_size * 2))
        if self.dilate_size > 0:
            self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate_size, self.dilate_size * 2))
        self.min_red = cv2.getTrackbarPos(str_MIN_RED, str_SETTINGS_W)

    def edges(self, _input):
        self.canny_min = cv2.getTrackbarPos(str_MIN_CANNY, str_SETTINGS_W)
        self.canny_min = cv2.getTrackbarPos(str_MAX_CANNY, str_SETTINGS_W)
        edges = cv2.Canny(_input, self.canny_min, self.canny_min)
        return edges

    def driving_lane(self, image):
        # function that finds the road driving lane line
        # image : camera image where the line locations are to be located
        # return : a masked image of onlt the lane lines
        self.min_lightness = cv2.getTrackbarPos(str_MIN_LIGHTNESS, str_SETTINGS_W)
        self.min_red = cv2.getTrackbarPos('R min', str_SETTINGS_W)
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

    @staticmethod
    def order_points(pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    @staticmethod
    def four_point_transform(_input, pts):
        rect = RoadSeeker.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32"
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(_input, M, (maxWidth, maxHeight))
        return warped

    def roiVertexes(self, _input, _focal_point=None, _source_pts=None, _roi_height=None):
        (_height, _width, _) = _input.shape
        if _focal_point is None:
            # default focal point is screen center
            _focal_point = self.focal_point
        if _source_pts is None:
            # default bottom points are at bottom screen corners
            _source_pts = [Point(0, _height), Point(_width, _height)]
        if _roi_height is None:
            # default roi height is a distance from bottom to horizon
            _roi_height = _focal_point.y + 10

        m_left = (_focal_point.y - _source_pts[0][1]) / (_focal_point.x - _source_pts[0].x)
        b_left = _focal_point.y - (m_left * _focal_point.x)
        x_left = (_roi_height - b_left) // m_left

        m_right = (_focal_point.y - _source_pts[1][1]) / (_focal_point.x - _source_pts[1].x)
        b_right = _focal_point.y - (m_right * _focal_point.x)
        x_right = (_roi_height - b_right) // m_right

        tl = Point(x_left, _roi_height)
        tr = Point(x_right, _roi_height)
        bl, fp, br = _source_pts[0], _focal_point, _source_pts[1]
        self.roi_region = [bl, fp, fp, br]
        points = np.array([[bl.x, bl.y],
                           [tl.x, tl.y],
                           [tr.x, tr.y],
                           [br.x, br.y]])
        return points

    def plotRoiBorder(self, _input):
        (_height, _width, _) = _input.shape
        points = self.roiVertexes(_input)
        color = (100,10,10)
        cv2.polylines(_input, np.int32([points]), 1, color, thickness=2)
        cv2.circle(_input, (self.focal_point.x, self.focal_point.y), 5, color, 2)



def process_video(input=0, mirror=False):
    processor = RoadSeeker()
    cam = cv2.VideoCapture(input)
    detector = CVD.ConvDetector(CVD._MODEL_FILENAME)

    while True:
        ret_val, original = cam.read()
        if not ret_val:
            print("No video frame captured: video at end or no video present.")
            quit()

        # First of all to increase perfomance we work with small video
        #raw = imutils.resize(original, _VIDEO_WIDTH)


        # hsl = processor.toHls(raw)
        # cv2.imshow('HSV', hsl)
        #
        # asphalt = processor.calcAsphaltColorsKmean(hsl)
        # # cv2.imshow('Asphalt', asphalt)
        #
        # # further processing for below-horizon part only to increase performance
        # # significant = processor.significant(hsv)
        #
        # thresholdedAnd = processor.thresholdedByAnd(hsl)
        # cv2.imshow('And Threshold', thresholdedAnd)
        #
        # # thresholdedOr = processor.thresholdedByOr(hsl)
        # # cv2.imshow('Or Threshold', thresholdedOr)
        #
        # thresholdedOnlyHue = processor.thresholdedOnlyHue(hsl)
        # cv2.imshow('Hue Threshold', thresholdedOnlyHue)
        #
        # cutted = processor.plotInfo(raw)
        # edges = processor.edges(thresholdedAnd)
        #
        # #cv2.imshow('Edges', processor.edges(edges))
        # cv2.imshow('Cutted', cutted)
        #
        # # cv2.imshow('Lane', processor.driving_lane(raw))

        #hls = processor.toHls(raw)

        # asphalt = processor.calcAsphaltColorsHistHLS(hls)

        #thresholdedOr = processor.thresholdedBy3Hist(hls)

        #cutted = processor.plotInfo(raw)
        #processor.plotRoiBorder(cutted)
        #cv2.imshow('Processed HSL', np.concatenate((cutted, thresholdedOr), axis=1))

        #asphalt = processor.calcAsphaltColorsKmean(raw)

        #thresholdedOr = processor.thresholdedByOr(raw)

        #cutted = processor.plotInfo(raw)
        #cv2.imshow('Processed RGB', np.concatenate((cutted, thresholdedOr), axis=1))

        #cv2.imshow('Lane', cv2.blur(thresholdedOr, (5,5)))

        dataForNN = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        nn_processed = detector.process(dataForNN)
        #cv2.imshow('Detector', nn_processed)

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()

if __name__ == "__main__":

    process_video('video/road6.mp4')
    # process_video()