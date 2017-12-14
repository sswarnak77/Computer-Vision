import cv2
import keras
from scipy.misc import imresize
import numpy as np


EMOTIONS =['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise','Neutral']

cascadeClassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
height = width = 20


def detect_face(image):
	faces = cascadeClassifier.detectMultiScale(image, 1.3, 5)
	for (x,y,w,h) in faces:
		image = image[y:y+h, x:x+w]
		return True,image
	return False,image


cap = cv2.VideoCapture(0)

modelWeights = keras.models.load_model('emotion_weightsGoogleNet.h5')

while cap.isOpened():
    ret, img = cap.read()
    if ret:
		x=[]
		gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
		flag,face = detect_face(gray)
		if flag:
			gray = imresize(face, [height, width], 'bilinear')
			gray = np.dstack((gray,) * 3)
			x.append(gray)
			x = np.asarray(x)
			print x.shape
			result=modelWeights.predict( x, batch_size=8, verbose=0)
			for index,emotion in enumerate(EMOTIONS):
				cv2.putText(img, emotion, (10,index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1);
				cv2.rectangle(img, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)
				print(result)

		cv2.imshow('TestingResult',img)
		if cv2.waitKey(1) &0xff==ord('q'):
			break
else:
	print "Can't open camera"

cv2.destroyAllWindows()