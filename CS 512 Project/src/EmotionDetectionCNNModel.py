from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
import numpy as np
import pandas as pd
from keras.utils import np_utils


#DATA PREPROCESSING

df = pd.read_csv("fer2013.csv")
df.head()

df.shape

df["Usage"].value_counts()

train = df[["emotion", "pixels"]][df["Usage"] == "Training"]
train.isnull().sum()
train['pixels'] = train['pixels'].apply(lambda im: np.fromstring(im, sep=' '))
x_train = np.vstack(train['pixels'].values)
y_train = np.array(train["emotion"])

x_train.shape, y_train.shape

public_test_df = df[["emotion", "pixels"]][df["Usage"]=="PublicTest"]
public_test_df["pixels"] = public_test_df["pixels"].apply(lambda im: np.fromstring(im, sep=' '))
x_test = np.vstack(public_test_df["pixels"].values)
y_test = np.array(public_test_df["emotion"])

x_train = x_train.reshape(-1, 48, 48, 1)
x_test = x_test.reshape(-1, 48, 48, 1)
x_train.shape, x_test.shape
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_train.shape, y_test.shape

batch_size = 128
nb_epoch = 50

print 'X_train shape: ', x_train.shape 
print 'y_train shape: ', y_train.shape
print 'img size: ', x_train.shape[2], x_train.shape[3]
print 'batch size: ', batch_size
print 'nb_epoch: ', nb_epoch


# model architecture:

model = Sequential()
model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu',input_shape=(48,48,1)))
model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

# optimizer:
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print 'Training....'
model.fit(x_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,validation_split=0.3, shuffle=True, verbose=1)

#Model result:
loss_and_metrics = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=1)
print 'Done!'
print 'Loss: ', loss_and_metrics[0]
print ' Acc: ', loss_and_metrics[1]


#Saving Model
json_string = model.to_json()
model.save_weights('emotion_weights.h5')
open('emotion_model.json', 'w').write(json_string)
model.save_weights('emotion_weights.h5')

