LETTERS = {
  "0": "А",
  "1": "Б",
  "2": "В",
  "3": "Г",
  "4": "Д",
  "5": "Е",
  "6": "Є",
  "7": "Ж",
  "8": "З",
  "9": "И",
  "10": "І",
  "11": "Й",
  "12": "К",
  "13": "Л",
  "14": "М",
  "15": "Н",
  "16": "О",
  "17": "П",
  "18": "Р",
  "19": "С",
  "20": "Т",
  "21": "У",
  "22": "Ф",
  "23": "Х",
  "24": "Ц",
  "25": "Ч",
  "26": "Ш",
  "27": "Ю",
  "28": "Я",
  "29": "Ь",
}
import os

import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(42,)))

model.add(tf.keras.layers.Dense(42, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(24, activation='relu'))
model.add(tf.keras.layers.Dense(30, activation='softmax'))

model.load_weights('program/Best_weights.h5')


from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='program/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)


import cv2
cap = cv2.VideoCapture(0)

while True:
    ret, frame= cap.read()
    # frame= cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(image)
    frame = draw_landmarks_on_image(frame, detection_result)
    if detection_result.hand_landmarks:
      row = []
      for landmark in detection_result.hand_landmarks[0]:
        row.append(landmark.x)
        row.append(landmark.y)
      y_pred = model(np.array([row])).numpy()
      y_classes = y_pred[0].argmax(axis=-1)
      letter = LETTERS[str(y_classes)]
      print(letter)


    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()