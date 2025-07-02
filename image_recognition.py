import numpy as np
import cv2

#step 1: recognize square
path = "test_image_3.jpg"
original = cv2.imread(path)
img = original.copy()
cv2.imshow("img",img)
area = img.shape[0]*img.shape[1]
print(area)

def order_points(pts):
    # Initialize list to be ordered: top left to bottom right
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect


def draw_contours(img, threshold_1 = 50, threshold_2 = 150, tolerance = 0.1, min_area = 0.2*area, showcontour = False):

    #magic card ratios
    width = 635
    height = 889
    magic_aspect = float(width)/height

    #image preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blurred, threshold_1, threshold_2)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #add some condition or loop here. we only want one rectangle/contour, the magic card. restrict conditions (area?)
    # if there are more, do nothing(?) if there are zero. should be easier once specifics of camera are known. 
    i = 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01*peri, True)

        if len(approx) == 4: #rectangle
            i+=1
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            area = cv2.contourArea(approx)
            if magic_aspect - tolerance <= aspect_ratio and magic_aspect + tolerance >= aspect_ratio:
                if area > min_area:
                    cv2.drawContours(img, [approx], -1, (255,255,0), 2)
    #from here, the function's purpose is changed to also include text recognition (for now, assume only one contour)
    verteces = approx.reshape(4,2).astype(np.float32)
    verteces = order_points(verteces)
    destination = np.array([[0,0], [width-1,0], [width-1, height-1], [0,height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(verteces, destination)
    warped = cv2.warpPerspective(original, M, (width, height))
    
    return warped

contoured = draw_contours(img)

cv2.imshow("test",contoured)
cv2.waitKey(0)
