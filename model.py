import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D
from keras.layers.core import Dropout

def get_img(source_path):
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    return image

def get_blurred_img(image):
    #gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.blur(image, (5, 5))

steering_offset = 0.1

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples = samples[1:]

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_angle = float(batch_sample[3])
                images.append(get_img(batch_sample[0]))
                angles.append(center_angle)
                images.append(get_img(batch_sample[1]))
                images.append(get_img(batch_sample[2]))
                angles.append(center_angle + 1.0)
                angles.append(center_angle - 1.0)
                #if center_angle > 0.0:
                #    angles.append(center_angle * 4.0)
                #    angles.append(0.0)
                #else:
                #    angles.append(0.0)
                #    angles.append(center_angle * 4.0)
            num_img = len(images)
            #for i in range(num_img):
            #    images.append(cv2.flip(images[i], 1))
            #    angles.append(-angles[i])
            num_img = len(images)
            for i in range(num_img):
                images.append(get_blurred_img(images[i]))
                angles.append(angles[i])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,4,4, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,4,4, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1000))
model.add(Dropout(p=0.5))
model.add(Dense(100))
model.add(Dropout(p=0.5))
model.add(Dense(10))
model.add(Dropout(p=0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=2)

model.save('model.h5')
print('Model saved')