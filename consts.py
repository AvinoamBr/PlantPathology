import  numpy as np
ORIGINAL_SIZE = (1365, 2048)
CENTER_CROP = (100,200)
NEW_SIZE = (np.array(ORIGINAL_SIZE)/12.5).astype(int)
# NEW_SIZE = (150, 150)
# NEW_SIZE = (64, 64)

# BATCH_SIZE = 128
BATCH_SIZE = 64
MODEL = 'VGG'
#PRETRAINED_VGG = r"C:\Users\User\PycharmProjects\PlantPathology\logs\fit\20200501-190352VGG\weights_epoch_23-loss_0.65.hdf5"
# MODEL = 'resnet50'


LEARNING_RATE = 0.00005
MINIMUM_LR = 0.0000000001