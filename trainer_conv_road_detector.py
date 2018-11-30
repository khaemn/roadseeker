# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.pyplot as plt


_MODEL_FILENAME = 'models/model_road_detector.h5'

_PLOT_BATCH = True

from PIL import Image
import os
import numpy as np

def __test__():
    return
    _model = load_model(_MODEL_FILENAME)

    imgdata = []
    imgdata.append(np.asarray(Image.open('train/generated/empty/_0_0_road2_5.jpg'), dtype='uint8'))
    imgdata.append(np.asarray(Image.open('train/generated/empty/_0_3_road1_39.jpg'), dtype='uint8'))
    imgdata.append(np.asarray(Image.open('train/generated/empty/_2_1_road1_15.jpg'), dtype='uint8'))
    imgdata.append(np.asarray(Image.open('train/generated/road/_3_5_road3_29.jpg'), dtype='uint8'))
    imgdata.append(np.asarray(Image.open('train/generated/road/_5_5_road6_44.jpg'), dtype='uint8'))
    imgdata.append(np.asarray(Image.open('train/generated/road/_0_5_road2_4.jpg'), dtype='uint8'))
    imgdata.append(np.asarray(Image.open('train/generated/road/_0_5_road2_6.jpg'), dtype='uint8'))
    imgdata.append(np.asarray(Image.open('train/generated/road/_0_5_road2_7.jpg'), dtype='uint8'))

    # img_true = Image.open('train/generated/road/_0_5_road3_3.jpg')
    # data1 = np.asarray(img_false
    # data2 = np.asarray(img_true, dtype='uint8')
    # try:
    # except SystemError:
    #     data = np.asarray(img.getdata(), dtype='uint8')

    # Adding an extra dimension for matcing input shape
    # data = np.expand_dims(data, axis=0)
    data = np.array(imgdata)

    print(_model.predict(data))

    #quit()





# dimensions of our images.
img_width, img_height = 100, 100

train_data_dir = 'train/generated'
validation_data_dir = 'train/validation'
nb_train_samples = 100 #00
nb_validation_samples = 10
epochs = 10
batch_size = 4  # 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    #horizontal_flip=False
)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

if _PLOT_BATCH:
    x_batch, y_batch = next(train_generator)
    for i in range (0, batch_size):
        image = x_batch[i]
        plt.imshow(image)
        plt.show()
    quit()


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              # optimizer='rmsprop',
              optimizer='adam',
              metrics=['accuracy'])

# Comment various thing here to make model train or saving from weights only
# model.load_weights('weights/road_trial_2.h5')
# model.save(_MODEL_FILENAME)
# quit()


for i in range(0, epochs):
    print("Iteration ", i, " of ", epochs)
    # __test__()
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=int(nb_validation_samples // batch_size),
        workers=8,
        verbose=2)
    model.save_weights('weights/road_trial_' + str(i) + '.h5')
    model.save(_MODEL_FILENAME)
    __test__()

