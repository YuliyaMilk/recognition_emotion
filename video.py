from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt

def convertMillis(millis):
    seconds, millis = divmod(millis, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return seconds, minutes, hours

# parameters for loading data and images
detection_model_path = 'cascades/haarcascade_frontalface_default.xml'
emotion_model_path = 'modelall.h5'

# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]
# EMOTIONS = ["happy", "sad", "neutral"]

df = pd.DataFrame(columns=['Person', 'Time', 'Emotion', 'Probability', 'Other emotion'])
name = 'Jhon'
last_emotion = ''
j = 0
count = 0
data_values = {
  "angry": 0,
  "disgust": 0,
  "scared": 0, 
  "happy": 0, 
  "sad": 0, 
  "surprised": 0, 
  "neutral": 0,
}

cap = cv2.VideoCapture('video_test.mp4')

while(cap.isOpened()):

  ret, frame = cap.read()
  if ret:
    
    frame = imutils.resize(frame,width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)


    frameClone = frame.copy()
    for (fX, fY, fW, fH) in faces:
      
      # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
      # the ROI for classification via the CNN
      roi = gray[fY:fY + fH, fX:fX + fW]
      roi = cv2.resize(roi, (48, 48))
      roi = roi.astype("float") / 255.0
      roi = img_to_array(roi)
      roi = np.expand_dims(roi, axis=0)
      
      
      preds = emotion_classifier.predict(roi)[0]
      emotion_probability = np.max(preds)
      label = EMOTIONS[preds.argmax()]
    
      cv2.putText(frameClone, label, (fX, fY - 10),
      cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
      cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                      (0, 0, 255), 2)
      if last_emotion:
        data_values[last_emotion] += 1
      if last_emotion!=label:
        last_emotion=label
        canvas = []
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                  text = "{}: {:.2f}%".format(emotion, prob * 100)
                  canvas.append(text)
        millis = cap.get(cv2.CAP_PROP_POS_MSEC)
        con_sec, con_min, con_hour = convertMillis(int(millis))
        time = "{0}:{1}:{2}".format(con_hour, con_min, con_sec)
        df.loc[j] = [name , time , label, round(emotion_probability*100, 2), canvas]
        j+=1

      #cv2.imshow('frame', frameClone)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  else: break

labels = []
sizes = []

for x, y in data_values.items():
    labels.append(x)
    sizes.append(y)

# Plot
patches, text = plt.pie(sizes)

plt.legend(patches, labels, loc='best')
plt.axis('equal')
plt.tight_layout()
plt.savefig('emotion_diagram.png')

with pd.ExcelWriter('teams1.xlsx', engine='xlsxwriter') as wb:
      df.to_excel(wb, sheet_name='Emotion', index=False)
      sheet = wb.sheets['Emotion']
      sheet.autofilter(0, 0, df.shape[0], 2)
      sheet.set_column('A:D', 10)
      sheet.set_column('E:E', 100)
      sheet.insert_image('G1', 'emotion_diagram.png')


cap.release()
cv2.destroyAllWindows()


