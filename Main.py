import cv2
import numpy as np
import operator
import os
import threading

from Information import Information
from OCREngine import OCREngine

def main():
    ocrEngine = OCREngine()
    threading.Thread(target = ocrEngine.startWebcam).start()

    while True:
        if ocrEngine.webcamImage is not None:
            image = ocrEngine.recognizeCharacters(ocrEngine.webcamImage)

            if ocrEngine.isRunning == False:
                break

if __name__ == "__main__":
    main()