import cv2
import numpy as np

toma = cv2.imread('D:/aaSDCNDJ/VehicleDetection/images/ejemplo1.JPG')

def draw_boxes(img, bboxes, color=(0,0,255),thick=6):
    # make a copy of the image
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn
    image = np.copy(img)
    for x1y1, x2y2 in bboxes:
        cv2.rectangle(image, x1y1, x2y2, color, thick)

    return image

boxes = [((275,502),(372,566)),((480,510),(546,557)),((588,508),(640,548)),((837,502),(1125,669))]

result = draw_boxes(toma, boxes)

cv2.imshow('Ventana',result)
cv2.waitKey(0)
cv2.destroyAllWindows()
