import cv2
import numpy as np

def draw_contours(img, threshold_1 = 50, threshold_2 = 150, tolerance = 0.1, min_area = 1000):

    magic_aspect = 6.35/8.89

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)



    edges = cv2.Canny(blurred, threshold_1, threshold_2)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01*peri, True)

        if len(approx) == 4: #rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            area = cv2.contourArea(approx)
            if magic_aspect - tolerance <= aspect_ratio and magic_aspect + tolerance >= aspect_ratio:
                if area > min_area:
                    cv2.drawContours(img, [approx], -1, (255,255,0), 2)
    return img

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        break
    
    contoured = draw_contours(frame)

    cv2.imshow("card scanner", contoured)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

