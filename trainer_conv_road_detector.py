# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.pyplot as plt
import random
#import convroaddetector as CVD

random.seed(777)

_MODEL_FILENAME = 'models/model_road_detector.h5'

_PLOT_BATCH = False


# dimensions of our images.
img_width, img_height = 100, 100

train_data_dir = 'train/generated'
validation_data_dir = 'train/validation'
nb_train_samples = 3000 #00
nb_validation_samples = 600
epochs = 5
batch_size = 32 # 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255
    , shear_range=0.051
    , zoom_range=0.051
    , horizontal_flip=True
    #horizontal_flip=False
)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
                                    rescale=1. / 255
                                  )

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
    , interpolation='bicubic'
    #, save_to_dir='generator_vis'
    #, save_format='jpg'
    , color_mode="rgb"

)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
    , color_mode="rgb"
)

if _PLOT_BATCH:
    x_batch, y_batch = next(train_generator)
    for i in range (0, batch_size):
        image = x_batch[i]
        plt.imshow(image)
        plt.show()
    #quit()


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# Initialising the CNN
model = Sequential()

model_init = "he_uniform"
#model_init = "glorot_normal"
#model_init = "uniform"

model.add(Conv2D(16, (5, 5), kernel_initializer=model_init, input_shape=input_shape, activation='relu', name="Conv2D_1"))
model.add(Conv2D(16, (5, 5), kernel_initializer=model_init, activation = 'relu', name="Conv2D_10"))
model.add(MaxPooling2D(pool_size=(8, 8)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(units=16, kernel_initializer=model_init, activation='relu'))
model.add(Dense(units=16, kernel_initializer=model_init, activation='relu'))
model.add(Dense(units=1, kernel_initializer="glorot_normal", activation='sigmoid'))
model.compile(optimizer='adam'
              , loss='binary_crossentropy'
              , metrics=['binary_accuracy']
              )

# Comment various thing here to make model train or saving from weights only
# model.load_weights('weights/road_trial_1.h5')
# model.save(_MODEL_FILENAME)
# CVD.__test__()
# quit()


for i in range(0, epochs):
    print("Iteration ", i, " of ", epochs)
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
