import matplotlib.pyplot as plt
import numpy as np
import keras
import keras.backend as k
import math
from get_data import unnormalize
from sklearn.metrics.classification import confusion_matrix as cfm
from sklearn.metrics import classification_report as cfr
from keras_preprocessing.image import  ImageDataGenerator

def show_batch(images,labels,columns_names):
    # get batch and display images and titles
    num_of_images = len(images)
    columns = math.ceil(math.sqrt(num_of_images))
    rows = math.ceil(num_of_images/columns)
    i = 0
    for row in range(rows):
        for col in range(columns):
            plt.subplot(rows,columns,i+1)
            plt.imshow(unnormalize(images[i]).astype(np.uint8))
            plt.axis('off')
            plt.title(columns_names[np.argmax(labels[i])])
            i += 1
            if i == num_of_images:
                plt.show()
                return

class confusion_matrix_callback(keras.callbacks.Callback):
    def __init__(self, train_data, valid_data):
        super(confusion_matrix_callback, self).__init__()
        self.predhis = []
        self.targets = []
        self.valid_data = valid_data
        self.train_data = train_data
        pass

    def index_to_label(self,y):
        label_names = self.valid_data.labels.columns
        labels_dict = {i:l for (i,l) in enumerate (label_names)}
        return [labels_dict[i] for i in y]

    def on_epoch_begin(self, epoch, logs=None):
        self.predhis = []
        self.targets = []

    def on_batch_end(self, batch, logs={}):
        x, y = self.train_data.X, self.train_data.y
        y_true = self.index_to_label(
            np.argmax(y,axis=1))
        self.targets.extend(y_true)
        y_pred = self.index_to_label(
            np.argmax(self.model.predict(x),axis=1))
        self.predhis.extend(y_pred)
        # print ("batch confusion matrix:")
        # print (cfm(y_true,y_pred,labels = self.train_data.labels.columns))

    def on_epoch_end(self, epoch, logs={}):
        print(f" ** epoch {epoch} confusion matrix train:")
        print(cfm( self.targets, self.predhis, labels=self.train_data.labels.columns))
        print(f" ** epoch {epoch} classification report train:")
        print(cfr( self.targets, self.predhis,
                  labels=self.valid_data.labels.columns,
                  target_names=self.valid_data.labels.columns))


        print(f" ** epoch {epoch} confusion matrix validation:")
        targets = []
        preds = []
        for valid_batch in self.valid_data:
            x, y = valid_batch
            y_true = self.index_to_label(
                np.argmax(y, axis=1))
            targets.extend(y_true)
            y_pred = self.index_to_label(
                np.argmax(self.model.predict(x), axis=1))
            preds.extend(y_pred)
            print (cfm(targets,preds))
            print (f" ** epoch {epoch} classification report validation :")
            print(cfr( targets, preds,
                      labels=self.valid_data.labels.columns,
                      target_names=self.valid_data.labels.columns ))



