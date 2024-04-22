import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Permute, Reshape, LayerNormalization, LSTM, Bidirectional, GaussianNoise

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

    
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

sys.path.append("../lib/")
from artifact_augmentation import *

tf.keras.backend.clear_session()

# TensorFlow wizardry
config = tf.compat.v1.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.98
# Create a session with the above options specified.
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))     

def ModelStructure(outptype):

    FrameSize = 50
    OverlapSize = 10 # actually, frame interval

    InpL = Input(shape=(3000,))

    InpFrame = tf.signal.frame(InpL, FrameSize, OverlapSize)
    InpFrameNoise = GaussianNoise(0.0)(InpFrame)

    Encoder = Dense(50, activation='relu', name = 'Encoder1')(InpFrameNoise)
    Encoder = Dropout(0.0)(Encoder, training=True)

    Encoder = Bidirectional(LSTM(25, return_sequences=True))(Encoder)
    Att_front = Bidirectional(LSTM(5, return_sequences=True))(Encoder)
    Att_front = LayerNormalization(axis=(1), epsilon=0.001)(Att_front)
    Att_front = Permute((2,1))(Att_front)
    Att_front = Dropout(0.0)(Att_front, training=True)
    Att_front = Dense(InpFrame.shape[1],activation='softmax')(Att_front)
    Att_front = Permute((2,1), name='Att_front')(Att_front)

    Context = InpFrameNoise[:,:,None] * Att_front[:,:,:,None]
    Context = tf.reduce_sum(Context, axis=(1), name = 'Context')

    Decoder = Bidirectional(LSTM(25, return_sequences=True))(Context)
    Decoder = LayerNormalization(axis=(1,2), epsilon=0.001)(Decoder)

    Att_back = Bidirectional(LSTM(250, return_sequences=False))(InpFrame)
    Att_back = Reshape((10, 50))(Att_back)
    Att_back = LayerNormalization(axis=(1,2), epsilon=0.001)(Att_back)
    Att_back = Bidirectional(LSTM(25, return_sequences=True))(Att_back)
    Att_back = Dropout(0.0)(Att_back, training=True)
    Att_back = Dense(50,activation='tanh')(Att_back)

    Scaling = Decoder + Att_back
    Scaling = Bidirectional(LSTM(25, return_sequences=True, name = 'Scaling'))(Scaling)

    ValDecoder = Context + Scaling
    ABPOut= Reshape((-1,),name='ABPOut')(ValDecoder)
    ABPOut_diff = Reshape((-1,),name='ABPOut_diff')(ABPOut[:,1:]-ABPOut[:,:-1])

    if (outptype == 1) or (outptype == 2):
        Casted = tf.cast(ABPOut, tf.complex64) 
        fft = tf.signal.fft(Casted)[:, :(Casted.shape[-1]//2+1)]
        AmplitudeOut = tf.abs(fft[:,1:])*(1/(fft.shape[-1])) 
        AmplitudeOut= Reshape((-1,),name='AmplitudeOut')(AmplitudeOut)

    if outptype == 0:
        AEModel = Model(InpL, (ABPOut, ABPOut_diff))
        loss_set = {'ABPOut':'mse', 'ABPOut_diff':RMSE}
        SaveFolder = '../models/DI_A/'
        SaveFilePath = 'DI_A_{epoch:04}_val{val_loss:.7f}_valABP{val_ABPOut_loss:.7f}_valABPd{val_ABPOut_diff_loss:.7f}_loss{loss:.7f}_ABP{ABPOut_loss:.7f}_ABPd{ABPOut_diff_loss:.7f}.hdf5'
    elif outptype == 1:
        AEModel = Model(InpL, (ABPOut, AmplitudeOut))
        loss_set = {'ABPOut':'mse', 'AmplitudeOut':AmplitudeCost}
        SaveFolder = '../models/DI_D/'
        SaveFilePath = 'DI_D_{epoch:04}_val{val_loss:.7f}_valABP{val_ABPOut_loss:.7f}_valAMP{val_AmplitudeOut_loss:.7f}_loss{loss:.7f}_ABP{ABPOut_loss:.7f}_AMP{AmplitudeOut_loss:.7f}.hdf5'
    elif outptype == 2:
        AEModel = Model(InpL, (ABPOut, ABPOut_diff, AmplitudeOut))
        loss_set = {'ABPOut':'mse', 'ABPOut_diff':RMSE, 'AmplitudeOut':AmplitudeCost}
        SaveFolder = '../models/DI/'
        SaveFilePath = 'DI_{epoch:04}_val{val_loss:.7f}_valABP{val_ABPOut_loss:.7f}_valABPd{val_ABPOut_diff_loss:.7f}_valAMP{val_AmplitudeOut_loss:.7f}_loss{loss:.7f}_ABP{ABPOut_loss:.7f}_ABPd{ABPOut_diff_loss:.7f}_AMP{AmplitudeOut_loss:.7f}.hdf5'    

    lrate = 0.0005
    decay = 1e-6
    adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=decay)
    AEModel.compile(loss=loss_set, optimizer=adam)

    return  AEModel, SaveFolder, SaveFilePath

