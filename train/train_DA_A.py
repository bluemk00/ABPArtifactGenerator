import os
import sys
import numpy as np
import tensorflow as tf

from ModelStructure_DA import *

sys.path.append("../lib/")
from artifact_augmentation import *

outptype = 0
'''
outptype
0 : (OutpData, OutpData_diff)               # DA-A
1 : (OutpData, OutAmplitude)                # DA-D
2 : (OutpData, OutpData_diff, OutAmplitude) # DA
'''

if __name__ == "__main__":

    BatchSize = 1500

    ## Data selection
    TrSet = np.load('../../../BioSignalCleaning/A.Data/A2.Processed/PPG/Train/MIMIC_PPG_TrSet.npy')[:]
    ValSet = np.load('../../../BioSignalCleaning/A.Data/A2.Processed/PPG/Train/MIMIC_PPG_ValSet.npy')[:]

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
