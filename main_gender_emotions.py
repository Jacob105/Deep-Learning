from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Load the pre-trained models
face_classifier = cv2.CascadeClassifier(r'C:\Users\jacob\Documents\Deep Learning\Modeller\haarcascade_frontalface_default.xml')
emotion_classifier = load_model(r'C:\Users\jacob\Documents\Deep Learning\Modeller\modelemotion.h5')
gender_classifier = load_model(r'C:\Users\jacob\Documents\Deep Learning\Modeller\modelgender.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Define gender labels
gender_labels = ['Female', 'Male']

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            # Emotion prediction
            roi_emotion = roi_gray.astype('float') / 255.0
            roi_emotion = img_to_array(roi_emotion)
            roi_emotion = np.expand_dims(roi_emotion, axis=0)

            emotion_prediction = emotion_classifier.predict(roi_emotion)[0]
            emotion_label = emotion_labels[emotion_prediction.argmax()]

            # Gender prediction
            roi_gender = cv2.resize(roi_gray, (48, 48))  # Resize to the input size of the gender model
            roi_gender = roi_gender / 255.0
            roi_gender = np.expand_dims(roi_gender, axis=0)
            roi_gender = np.expand_dims(roi_gender, axis=-1)

            gender_prediction = gender_classifier.predict(roi_gender)[0]
            gender_label = gender_labels[int(gender_prediction > 0.5)]  # Assuming binary classification

            # Display the emotion and gender labels
            label_position = (x, y)
            cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Gender: {gender_label}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion and Gender Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

