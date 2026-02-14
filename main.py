import mediapipe as mp
import numpy as np
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


import urllib.request

# URL for the hand landmarker model
model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
model_path = "hand_landmarker.task"

# Download the file
urllib.request.urlretrieve(model_url, model_path)
print("Model downloaded successfully!")


img = cv2.imread("./images/woman_hand.jpg")
cv2.imshow("Image", img)
cv2.waitKey(0)  # (this is necessary to avoid Python kernel form crashing)
cv2.destroyAllWindows()  # closing all open windows


# Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

detection_result = detector.detect(mp_image)
