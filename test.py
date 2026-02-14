import mediapipe as mp
import numpy as np
import cv2

img = cv2.imread("./images/woman_hand.jpg")
cv2.imshow("Image", img)
cv2.waitKey(0)  # (this is necessary to avoid Python kernel form crashing)
cv2.destroyAllWindows()  # closing all open windows
