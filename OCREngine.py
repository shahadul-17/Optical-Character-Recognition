import cv2
import numpy as np
import operator
import os

from Rectangle import Rectangle
from Information import Information

class OCREngine:

    dodgerBlueColor = None
    classifications = None
    flattenedImages = None
    webcam = None
    kNearest = None

    def __init__(self):
        self.rectangles = []        # list of valid rectangles...
        self.isRunning = False
        self.text = ""
        self.webcamImage = None

        self.RESIZED_IMAGE_WIDTH = 20
        self.RESIZED_IMAGE_HEIGHT = 30

        global webcam, classifications, flattenedImages, kNearest, dodgerBlueColor

        dodgerBlueColor = (255, 144, 30)
        webcam = cv2.VideoCapture(0)
        classifications = np.loadtxt("data\\classifications.txt", np.float32)
        classifications = classifications.reshape((classifications.size, 1))    # reshape numpy array to 1d, necessary to pass to call to train
        flattenedImages = np.loadtxt("data\\flattened-images.txt", np.float32)

        kNearest = cv2.ml.KNearest_create()
        kNearest.train(flattenedImages, cv2.ml.ROW_SAMPLE, classifications)

    def showImage(self, image):
        cv2.imshow("", image)
        
        if cv2.waitKey(1) == 27:
            self.stopWebcam()

    def recognizeCharacters(self, image):
        global text, kNearest

        text = ""
        informations = []
        validInformations = []

        imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)       # get grayscale image
        imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur

                                                            # filter image from grayscale to black and white
        imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                        255,                                  # make pixels that pass the threshold full white
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                        cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                        11,                                   # size of a pixel neighborhood used to calculate threshold value
                                        2)                                    # constant subtracted from the mean or weighted mean

        imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image

        imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                    cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                    cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

        for contour in contours:
            information = Information()
            information.contour = contour
            information.boundingRectangle = cv2.boundingRect(information.contour)
            information.calculateRectangle()
            information.contourArea = cv2.contourArea(information.contour)
            informations.append(information)
        
        for information in informations:
            if information.isContourValid():
                validInformations.append(information)
        
        validInformations.sort(key = operator.attrgetter("rectangle.x"))         # sort contours from left to right

        for information in validInformations:
            # cv2.rectangle(image, (information.x, information.y), (information.x + information.width, information.y + information.height), (0, 255, 0), 2)

            imgROI = imgThresh[information.rectangle.y:information.rectangle.y + information.rectangle.height, information.rectangle.x : information.rectangle.x + information.rectangle.width]

            imgROIResized = cv2.resize(imgROI, (self.RESIZED_IMAGE_WIDTH, self.RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

            npaROIResized = imgROIResized.reshape((1, self.RESIZED_IMAGE_WIDTH * self.RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

            npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

            retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)

            if dists > 4712875:
                continue
            
            self.rectangles.append(information.rectangle)

            print "=" + str(dists) + "="
            
            text = text + str(chr(int(npaResults[0][0])))            # append current char to full string
        
        if len(text.strip()) != 0:
            print text + "\n"

        return image

    def startWebcam(self):
        self.isRunning = True

        global dodgerBlueColor

        while self.isRunning:
            returnValue, self.webcamImage = webcam.read(0)

            for rectangle in self.rectangles:
                # if self.x != 0 and self.y != 0 and self.width != 0 and self.height !=0:
                cv2.rectangle(self.webcamImage, (rectangle.x, rectangle.y), (rectangle.x + rectangle.width, rectangle.y + rectangle.height), dodgerBlueColor, 2)   # marks characters...
                self.rectangles.remove(rectangle)
            
            self.showImage(self.webcamImage)
    
    def stopWebcam(self):
        self.isRunning = False
        cv2.destroyAllWindows()