import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Load emotion model (no compile to avoid keras version issues)
model = load_model("fer2013_mini_XCEPTION.102-0.66.hdf5", compile=False)

# Emotion labels
emotion_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

# Buffer to smooth predictions
emotion_window = deque(maxlen=10)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]

        try:
            # Resize to match model input
            face = cv2.resize(face,(64,64))
        except:
            continue

        # Normalize
        face = face / 255.0

        # Add batch + channel dimension
        face = np.reshape(face,(1,64,64,1))

        # Predict emotion
        preds = model.predict(face, verbose=0)[0]

        emotion_probability = np.max(preds)
        label = emotion_labels[preds.argmax()]

        # Add to smoothing buffer
        emotion_window.append(label)

        try:
            label = max(set(emotion_window), key=emotion_window.count)
        except:
            pass

        # Draw face box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)

        # Show main emotion
        text = f"{label} ({emotion_probability*100:.0f}%)"
        cv2.putText(frame,text,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

        # Emotion bars
        for i,(emotion,prob) in enumerate(zip(emotion_labels,preds)):

            text = f"{emotion}: {prob*100:.0f}%"
            bar = int(prob*300)

            cv2.rectangle(frame,(10,i*30+10),(bar,i*30+30),(255,0,255),-1)

            cv2.putText(frame,text,(10,i*30+25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    cv2.imshow("Face Emotion Detector",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
