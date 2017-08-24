from Rectangle import Rectangle

class Information:

    def __init__(self):
        self.MINIMUM_CONTOUR_AREA = 100
        self.rectangle = Rectangle()
        contourArea = 0.0
        contour = None
        boundingRectangle = None

    def calculateRectangle(self):
        [x, y, width, height] = self.boundingRectangle
        
        self.rectangle.x = x
        self.rectangle.y = y
        self.rectangle.width = width
        self.rectangle.height = height

    def isContourValid(self):
        if self.contourArea < self.MINIMUM_CONTOUR_AREA:
            return False
        
        return True