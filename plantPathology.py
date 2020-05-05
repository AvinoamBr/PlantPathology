import os
import datetime
import keras
import pandas as pd

from get_data import  get_generators, get_labels, get_class_weights
from model import model
from utils import show_batch, confusion_matrix_callback
import consts as C

if __name__ == "__main__":
    labels_columns = get_labels().columns
    train_generator, validation_generator = get_generators()
    # imgs, labels = train_generator[0]
    # show_batch(imgs,labels,labels_columns)

    print ("start training")


    #  -- callbacks --
    log_dir=r"C:\Users\User\PycharmProjects\PlantPathology\logs\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + C.MODEL
    os.mkdir(log_dir)


    csv_logger = keras.callbacks.CSVLogger(log_dir + '\\training.log')
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=3, min_lr=C.MINIMUM_LR, verbose = True)
    modelCP= keras.callbacks.callbacks.ModelCheckpoint(
        log_dir + "\\weights_epoch_{epoch:02d}-loss_{val_loss:.2f}.hdf5",
        monitor='val_loss', verbose=0, save_best_only=False,
        save_weights_only=False, mode='auto', period=1)
    # lr_sched = keras.callbacks.LearningRateScheduler(lambda epoch: 0.002* 0.75 ** (epoch-1) )
    cfm_callback = confusion_matrix_callback(train_generator, validation_generator)
    callbacks = [tensorboard_callback, reduce_lr, modelCP, csv_logger]# , lr_sched] #, cfm_callback ]
    # ------------------------------

    # saved_weights = r"C:\Users\User\PycharmProjects\PlantPathology\logs\fit\20200331-120037\weights_epoch_17-loss_1.00.hdf5"
    # saved_weights = r"C:\Users\User\PycharmProjects\PlantPathology\logs\fit\saved_weights\weights_epoch_34-loss_1.06.hdf5"
    if C.MODEL == 'VGG' and C.PRETRAINED_VGG:
        model.load_weights(C.PRETRAINED_VGG)

    model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        epochs=500,
        callbacks=callbacks,
        verbose= True,
        # class_weight=get_class_weights()
        class_weight= 1/pd.Series(train_generator.classes).value_counts(),
        # initial_epoch = 20,
        # class_weight= 'auto',
    )




