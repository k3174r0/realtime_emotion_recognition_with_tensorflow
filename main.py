import cv2
import numpy as np
#from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

classes = ({0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'})
cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml") #you should edit.
cap = cv2.VideoCapture(0)

model_path = '../face_classification/trained_models/fer2013_mini_XCEPTION.119-0.65.hdf5'
emotions_XCEPTION = load_model(model_path, compile=False)

while(True):
    ret, frame = cap.read()
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facerect = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    if len(facerect)>0:
        for rect in facerect:
            gray = gray[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            try:
                gray = cv2.resize(gray, (48, 48))
            except:
                break
            img_array = image.img_to_array(gray)
            pImg = np.expand_dims(img_array, axis=0)/255
            cv2.rectangle(frame, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), (255,255,255), thickness=1)
            prediction = emotions_XCEPTION.predict(pImg)[0]
            top_indices = prediction.argsort()[-5:][::-1]
            result = [(classes[i],prediction[i])for i in top_indices]
            cv2.putText(frame, result[0][0], tuple(rect[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), thickness=2)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
