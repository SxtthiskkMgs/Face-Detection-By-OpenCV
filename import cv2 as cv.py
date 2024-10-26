import cv2
import numpy as np
import sqlite3
import os
from datetime import datetime

length_boundingbox = 30
thick_boundingbox = 5

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the LBPH face recognizer and load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model/trainingData.yml")  # Load trained face recognition model

# Connect to SQLite database
conn = sqlite3.connect('C:/Users/oniko/Downloads/04_Resource_Face_Recognition_Final/database.db')
cons = conn.cursor()

# Path to the image for face recognition
img_path = 'C:/Users/oniko/OneDrive/Desktop/Face detectopm/images'

# Read the image and convert to grayscale
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

for (x, y, w, h) in faces:
    # Draw rectangle around the detected face
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    # Draw bounding box lines
    x1, y1 = x + w, y + h
    cv2.line(img, (x, y), (x + length_boundingbox, y), (0, 255, 0), thick_boundingbox)
    cv2.line(img, (x, y), (x, y + length_boundingbox), (0, 255, 0), thick_boundingbox)
    cv2.line(img, (x1, y), (x1 - length_boundingbox, y), (0, 255, 0), thick_boundingbox)
    cv2.line(img, (x1, y), (x1, y + length_boundingbox), (0, 255, 0), thick_boundingbox)
    cv2.line(img, (x, y1), (x + length_boundingbox, y1), (0, 255, 0), thick_boundingbox)
    cv2.line(img, (x, y1), (x, y1 - length_boundingbox), (0, 255, 0), thick_boundingbox)
    cv2.line(img, (x1, y1), (x1 - length_boundingbox, y1), (0, 255, 0), thick_boundingbox)
    cv2.line(img, (x1, y1), (x1, y1 - length_boundingbox), (0, 255, 0), thick_boundingbox)

    # Recognize the face
    ids, conf = recognizer.predict(gray[y: y + h, x: x + w])
    
    # Fetch the recognized name from the SQLite database
    cons.execute("SELECT name FROM users WHERE id = (?)", (ids,))
    result = cons.fetchall()

    if result:
        name = result[0][0]
    else:
        name = "Unknown"

    if conf < 50:
        print(f"Face Recognized: {name} with Confidence: {conf}")
    else:
        print("Face not recognized")

# Display the image with bounding boxes
cv2.imshow("Image with Bounding Boxes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Close the SQLite connection
conn.close()
