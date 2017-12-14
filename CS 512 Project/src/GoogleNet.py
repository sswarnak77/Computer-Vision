from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.layers.containers import Graph
from keras.models import Sequential

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

model = Graph()

model.add_input(name='n00', input_shape=(1,48,48))

# layer 1
model.add_node(Convolution2D(64,1,1, activation='relu'), name='n11', input='n00')
model.add_node(Flatten(), name='n11_f', input='n11')

model.add_node(Convolution2D(96,1,1, activation='relu'), name='n12', input='n00')

model.add_node(Convolution2D(16,1,1, activation='relu'), name='n13', input='n00')

model.add_node(MaxPooling2D((3,3),strides=(2,2)), name='n14', input='n00')

# layer 2
model.add_node(Convolution2D(128,3,3, activation='relu'), name='n22', input='n12')
model.add_node(Flatten(), name='n22_f', input='n22')

model.add_node(Convolution2D(32,5,5, activation='relu'), name='n23', input='n13')
model.add_node(Flatten(), name='n23_f', input='n23')

model.add_node(Convolution2D(32,1,1, activation='relu'), name='n24', input='n14')
model.add_node(Flatten(), name='n24_f', input='n24')

# output layer
model.add_node(Dense(1024, activation='relu'), name='layer4',
               inputs=['n11_f', 'n22_f', 'n23_f', 'n24_f'], merge_mode='concat')
model.add_node(Dense(10, activation='softmax'), name='layer5', input='layer4')
model.add_output(name='output1',input='layer5')


print 'Training....'
model.compile(loss={'output1':'categorical_crossentropy'}, optimizer='adam',metrics=['accuracy'])
model.fit({'n00':x_train, 'output1':y_train}, nb_epoch=nb_epoch, batch_size=batch_size,validation_split=0.3, shuffle=True, verbose=1)


#Model result:
loss_and_metrics = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=1)
print 'Done!'
print 'Loss: ', loss_and_metrics[0]
print ' Acc: ', loss_and_metrics[1]


#Saving Model
json_string = model.to_json()
model.save_weights('emotion_weights_googleNet.h5')
open('emotion_model_googleNet.json', 'w').write(json_string)
model.save_weights('emotion_weights_googleNet.h5')

