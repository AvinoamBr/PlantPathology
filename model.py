import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from keras import regularizers
import consts as C
from consts import  NEW_SIZE, MODEL
from keras import applications


pretrained_model = None
if MODEL == 'resnet50':
    pretrained_model = applications.ResNet50(weights = C.PRETRAINED_VGG or 'imagenet', include_top = False, input_shape = (*NEW_SIZE, 3))

if MODEL == 'VGG':
    pretrained_model = applications.VGG19(weights ='imagenet', include_top = False, input_shape = (*NEW_SIZE, 3))

if pretrained_model:
    # for l in pretrained_model.layers[:5]:
    #     l.trainable = False

    # trainable = True is default..
    # however, implicit it just for clearness.
    for l in pretrained_model.layers:
        l.trainable = True

    top_model = Sequential([Flatten(),
                            Dropout(0.1),
                            Dense(100,activation='relu'),
                            Dropout(0.25),
                            Dense(4, activation='softmax')])

    model = Sequential()
    model.add(pretrained_model)
    model.add(top_model)




model.compile(
    optimizer= keras.optimizers.Adam(lr = C.LEARNING_RATE ),
    loss='categorical_crossentropy',
    metrics=['accuracy'] #, confusion_matrix_metrics],
    )

