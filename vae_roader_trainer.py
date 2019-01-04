# https://blog.keras.io/building-autoencoders-in-keras.html

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation, Dropout, Flatten
from keras.models import Model
import random
import os
import cv2
import numpy as np

random.seed(777)

_MODEL_FILENAME = 'models/model_vae_roader.h5'

# dimensions of our images.
img_width, img_height = 160, 90

train_data_dir = 'dataset/train/X'
validation_data_dir = 'dataset/validation/X'
nb_train_samples = 200
nb_validation_samples = 20
epochs = 20
iterations = 10
batch_size = 32

# TODO: Data loading should be reworked to keras Generators to improve perf.
def load_data(path=train_data_dir):
    X, Y = [], []
    for root_back, dirs_back, files_back in os.walk(path):
        for _file in files_back:
            image = cv2.imread(os.path.join(path, _file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            _x = image[:, :img_width]
            _y = image[:, img_width:]
            _x = _x.astype('float32')
            _x = _x / 255
            X.append(_x)

            # Note: origin resolution is 160*90 however upsampling KERAS layers
            # produces 160*96 (as far both numbers are divided by 4 which is UpSampling 2*2
            # coeffitient. I wa too lazy to overcome this, so here the Y data is just adjusted to necessary
            # resolution produced by UpSampling layers.
            _y = cv2.resize(_y, (160, 96))
            _y = cv2.cvtColor(_y, cv2.COLOR_RGB2GRAY)
            _y = _y.astype('float32')
            _y = _y / 255
            _y = _y.reshape((96, 160, 1))
            Y.append(_y)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

# Creating the Model

input_img = Input(shape=(img_height, img_width, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same', name='Encoder_CONV2D_1')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='Encoder_CONV2D_2')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Encoder_CONV2D_3')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='Encoder_CONV2D_4')(x)

encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (6, 10, 128)

x = Conv2D(128, (3, 3), activation='relu', padding='same', name='Decoder_CONV2D_1')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Decoder_CONV2D_3')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='Decoder_CONV2D_4')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='Decoder_CONV2D_5')(x)
x = UpSampling2D((2, 2))(x)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',
                    metrics=['accuracy'])

print("AE info:", autoencoder.summary())

_TRAIN_ME = True
if _TRAIN_ME:
    X, Y = load_data()
    for i in range(0, iterations):
        print("Iteration ", i, " of ", epochs)
        autoencoder.fit(X, Y,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.1)
        autoencoder.save_weights('weights/roader_trial_' + str(i) + '.h5')
        autoencoder.save(_MODEL_FILENAME)
else:
    autoencoder.load_weights('weights/roader_trial_1.h5')


# TEST:
path = 'dataset/test'
test_X = []
test_images = []
for root_back, dirs_back, files_back in os.walk(path):
    for _file in files_back:
        image = cv2.imread(os.path.join(path, _file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        test_images.append(image)

        image = cv2.resize(image, (img_width, img_height))
        image = image / 255
        test_X.append(image)

test_X = np.array(test_X)
predicted = autoencoder.predict(test_X)
# denormalize
predicted = predicted * 255

for i in range(0, len(test_images)):
    output = predicted[i].astype('uint8')
    _input = test_images[i]
    (height, width, _) = _input.shape
    _input = cv2.cvtColor(_input, cv2.COLOR_RGB2BGR)
    processed = cv2.resize(output, (width, height), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Input', _input)
    cv2.imshow('Output', processed)

    alpha = 0.2
    _, mask = cv2.threshold(processed, 200, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    combined = np.array(_input, dtype=np.uint8)
    color_fill = _input
    color_fill[:, :] = [255, 50, 255]
    color_fill = cv2.bitwise_and(color_fill, color_fill, mask=mask)
    cv2.addWeighted(combined, 1 - alpha, color_fill, alpha, 0, combined)
    cv2.imshow('Prediction', combined)

    cv2.waitKey(0)
cv2.destroyAllWindows()

