class Information:

    def __init__(self):
        x = 0
        y = 0
        width = 0
        height = 0
        contourArea = 0.0
        self.MINIMUM_CONTOUR_AREA = 100
        contour = None
        rectangle = None

    def calculateRectangle(self):
        [x, y, width, height] = self.rectangle
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def isContourValid(self):
        if self.contourArea < self.MINIMUM_CONTOUR_AREA:
            return False
        
        return True