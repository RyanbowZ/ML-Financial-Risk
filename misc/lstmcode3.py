import keras.backend as K
from keras.layers import Multiply
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, merge
from keras.layers import Input, Dense, Masking
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.preprocessing import sequence
from keras import optimizers
from keras.utils.vis_utils import plot_model
import pandas as pd

SINGLE_ATTENTION_VECTOR = False

def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
#    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
#        if print_shape_only:
#            print(layer_activations.shape)
#        else:
#            print(layer_activations)
    return activations

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    #a = Permute((2, 1))(inputs)
    #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)
    #a_probs = a
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


f=pd.read_csv("D:\Documents\毕业设计\code\qq44lstm.csv",index_col="trade_date")

ss=[]
for i in np.unique(f.index):
    s=np.array(f.loc[i])
    if s.ndim==1:
        ss.append([s.tolist()])

    else:
        ss.append(s.tolist())

seq=sequence.pad_sequences(np.array(ss),maxlen=6,dtype='float',padding='post')

# reshape input into [samples, timesteps, features]
n_sample=seq.shape[0]
n_feature = seq.shape[2]
n_timestep = seq.shape[1]

#lstm_1=LSTM(100, activation='relu', input_dim=n_feature)
inputs=Input(shape=(n_timestep,n_feature,))
#inputs=LSTM(100, activation='relu', input_dim=n_feature)

lstm_out= LSTM(100, activation='relu',return_sequences=True)(inputs)
attention_mul = attention_3d_block(lstm_out)
attention_mul = Flatten()(attention_mul)
output = Dense(n_feature, activation='sigmoid')(attention_mul)
lstm_2=LSTM(100, activation='relu', return_sequences=True)(output)
outp2=TimeDistributed(Dense(n_feature))(lstm_2)
model = Model(inputs=[inputs], outputs=outp2)
'''model = Sequential()

model.add(LSTM(100, activation='relu', input_dim=n_feature))
model.add(RepeatVector(1))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(attention_3d_block())
model.add(TimeDistributed(Dense(n_feature)))'''


model.compile(optimizer='adam', loss='mse')
model.summary()
model.fit(seq, seq, epochs=10, verbose=1)

# demonstrate recreation
yhat = model.predict(seq, verbose=1)

yhat2=np.reshape(yhat,(-1,n_feature))

print(yhat2)

attention_vector = get_activations(model, seq[0],
                                       print_shape_only=True,
                                       layer_name='attention_vec')[0].flatten()
print('attention =', attention_vector)

pd.DataFrame(yhat2).to_csv('yhat_attention.csv')