import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def pickPoint(img, nZoomIteration = 0, maxW=800, maxH=500):
    x, y, x0, y0 = 0, 0, 0, 0
    x1, y1 = img.shape[1], img.shape[0] 
    winName = "Select roi. Press ESC to use last pick."
    # first time
    #   Select scale
    if nZoomIteration <= 0:
        nZoomIteration = 999
    for i in range(nZoomIteration):
        scale_x = maxW * 1.0 / (x1 - x0)
        scale_y = maxH * 1.0 / (y1 - y0)
        scale_x = min(scale_x, scale_y)
        scale_y = scale_x
        #   new image
        imgshow = cv.resize(img[y0:y1,x0:x1], dsize=(-1,-1), fx=scale_x, fy=scale_y, \
                            interpolation= cv.INTER_LANCZOS4)
        scale_x = imgshow.shape[1] * 1.0 / (x1 - x0)
        scale_y = imgshow.shape[0] * 1.0 / (y1 - y0)
        #   get point
        roi_x, roi_y, roi_w, roi_h = cv.selectROI( \
            winName, imgshow, showCrosshair=True, fromCenter=False)
        if (roi_w == 0 or roi_h == 0): 
            cv.destroyWindow(winName)
            break
        #   calculate position (not sure why needs to add -0.5 at the end)
#        x = (x0 - 0.5 + 0.5 / scale_x) + (roi_x + 0.5 * (roi_w - 1.)) / scale_x 
#        y = (y0 - 0.5 + 0.5 / scale_y) + (roi_y + 0.5 * (roi_h - 1.)) / scale_y
        x = x0 - 0.5 + (roi_x + 0.5 * roi_w) / scale_x 
        y = y0 - 0.5 + (roi_y + 0.5 * roi_h) / scale_y 
#        print("This: ", "; x=", x, ";x0=", x0, ";x1=", x1, ";roi_x=", roi_x, ";roi_w=", roi_w, ";scale_x=", scale_x)
        x0 = int(x - ((roi_w - 1) * 0.5) / scale_x + 0.5)
        y0 = int(y - ((roi_h - 1) * 0.5) / scale_y + 0.5)
        x1 = int(x + ((roi_w - 1) * 0.5) / scale_x + 0.5) + 1
        y1 = int(y + ((roi_h - 1) * 0.5) / scale_y + 0.5) + 1
#        print("Next: ", "; x=", x, ";x0=", x0, ";x1=", x1, ";roi_x=", roi_x, ";roi_w=", roi_w, ";scale_x=", scale_x)
        cv.destroyWindow(winName)
    return [x, y, x0, y0, (x1 - x0), (y1 - y0)]

def drawMarkerShift(img, position, color = 128, markerType = cv.MARKER_CROSS, \
                    markerSize = 20, thickness = 2, line_type = 8, \
                    shift = 0):
    # imgMarked = img.copy(); cv.drawMarker(imgMarked, (179,179), 128, cv.MARKER_CROSS, markerSize=40, thickness = 3); plt.imshow(imgMarked)
    shift = int(shift + 0.5)
    if shift < 0:
        shift = 0
    if shift > 3:
        shift = 3
    scale = 2 ** shift
    imgResized = np.zeros((img.shape[0] * scale, img.shape[1] * scale), \
                           img.dtype)
    imgResized = cv.resize(img, (-1,-1), imgResized, fx = scale, fy = scale, \
                           interpolation = cv.INTER_CUBIC)
    positionResized = position.copy()
    positionResized[0] = int((position[0] + 0.5) * scale - 0.5 + 0.5)
    positionResized[1] = int((position[1] + 0.5) * scale - 0.5 + 0.5)
    markerSizeResized = int(markerSize * scale + 0.5)
    thicknessResized = int(thickness * scale + 0.5)
    imgResized = cv.drawMarker(imgResized, positionResized, color, markerType, \
                  markerSizeResized, \
                  thicknessResized, line_type)
    img = cv.resize(imgResized, (-1,-1), img, fx = 1./scale, \
                    fy = 1./scale, interpolation = cv.INTER_LANCZOS4)
#    plt.imshow(img, cmap='gray')
    return img
    

def pickTmGivenPoint(img, pt, markColor= [], markerType = cv.MARKER_CROSS, \
                 markerSize = 20, markerThickness = 2, markerLineType = 8, \
                 markerShift = 1):
    # draw pt on img
    imgClone = img.copy()
    if (len(markColor) == 0):
        if (len(imgClone.shape) == 2): # gray scale
            markColor = 255
        else:
            markColor = [64,255,64]
    imgClone = drawMarkerShift(imgClone, pt, markColor, markerType, \
                               markerSize, markerThickness, \
                               shift = markerShift)
    ptAndBox = pickPoint(imgClone, 1)
    return ptAndBox[2:]

def pickPointAndTm(img):
    ptAndBox = pickPoint(img)
    tmX0, tmY0, tmW, tmH = pickTmGivenPoint(img, ptAndBox[0:2])
    return [ptAndBox[0], ptAndBox[1], tmX0, tmY0, tmW, tmH]

def pickPoint_example(imgSize = 360, imgRot = 30.0):
    # make sure imgSize is an even
    imgSize = int(imgSize/2) * 2
    # generate image with a black-white rectangular corner
    img = np.zeros((imgSize, imgSize), dtype = np.uint8)
    img[0:int(imgSize/2), 0:int(imgSize/2)] = 200;
    img[int(imgSize/2):imgSize, int(imgSize/2):imgSize] = 200;
    # cv.imshow("TETT", img); cv.waitKey(0);
    # cv.destroyWindow("TETT")
    (px,py,x0,y0,w,h) = pickPoint(img)
    print((px,py,x0,y0,w,h))
    
def pickPointAndTm_example(imgSize = 360, imgRot = 30.0):
    # make sure imgSize is an even
    imgSize = int(imgSize/2) * 2
    # generate image with a black-white rectangular corner
    img = np.zeros((imgSize, imgSize), dtype = np.uint8)
    img[0:int(imgSize/2), 0:int(imgSize/2)] = 200;
    img[int(imgSize/2):imgSize, int(imgSize/2):imgSize] = 200;
    # cv.imshow("TETT", img); cv.waitKey(0);
    # cv.destroyWindow("TETT")
    (px,py,x0,y0,w,h) = pickPointAndTm(img)
    print((px,py,x0,y0,w,h))
    plt.imshow(img[x0:x0+w,y0:y0+h], cmap='gray')
    
