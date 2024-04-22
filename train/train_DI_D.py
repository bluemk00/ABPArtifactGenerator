import os
import sys
import numpy as np
import tensorflow as tf

from ModelStructure import *

sys.path.append("../lib/")
from artifact_augmentation import *

outptype = 1
'''
outptype
0 : (OutpData, OutpData_diff)               # DI-A
1 : (OutpData, OutAmplitude)                # DI-D
2 : (OutpData, OutpData_diff, OutAmplitude) # DI
'''

if __name__ == "__main__":

    BatchSize = 1500

    ## Data selection
    TrSet = np.load('../../../BioSignalCleaning/A.Data/A2.Processed/ABP/Train/MIMIC_ART_TrSet.npy')[:]
    ValSet = np.load('../../../BioSignalCleaning/A.Data/A2.Processed/ABP/Train/MIMIC_ART_ValSet.npy')[:]
    TrSet = (TrSet - 20.0) / (220.0 - 20.0)
    ValSet = (ValSet - 20.0) / (220.0 - 20.0)

    print('************************************************')
    print(f'    Train set total {TrSet.shape[0]} size   ')
    print(f'    Valid set total {ValSet.shape[0]} size   ')
    print('************************************************')
    
    strategy = tf.distribute.MirroredStrategy( cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) 

    # Data Loader
    TrainSet = DataBatch(TrSet[:], BatchSize, outptype=outptype)
    ValidSet = DataBatch(ValSet[:], BatchSize, outptype=outptype)

    AEModel, SaveFolder, SaveFilePath = ModelStructure(outptype)
    
    checkpoint = ModelCheckpoint(SaveFolder + SaveFilePath, monitor=('val_loss'), verbose=0, save_best_only=True, mode='auto', period=1) 
    earlystopper = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True)
    history = LossHistory(SaveFolder + 'training_loss.csv')

    AEModel.fit(TrainSet, validation_data = (ValidSet), verbose=1, epochs=100, callbacks=[history,earlystopper,checkpoint])