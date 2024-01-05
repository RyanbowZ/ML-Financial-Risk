
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Masking
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.preprocessing import sequence
from keras import optimizers
from keras.utils.vis_utils import plot_model
import pandas as pd
f=pd.read_csv("D:\Documents\毕业设计\code\qq4lstm.csv",index_col="trade_date")
#lT=0
#cn=0
#mts=0
#ts=0
'''
seq=[]
for i in np.unique(f.index):
    #print(i)
    s=np.array(f.loc[i])
    if s.ndim==1:
        seq.append(s.tolist())
        for j in range(11):
            seq.append([0,0,0,0,0,0,0,0,0,0,0])
    else:
        #ts=0
        for ss in s:
            seq.append(ss.tolist())
        for j in range(12-len(s)):
            seq.append([0, 0, 0, 0, 0, 0,0,0,0,0,0])
            #ts+=1
    cn+=1

    #mts=max(mts,ts)
#print(seq)
seq=np.array(seq)'''
#print(f.values)


ss=[]
for i in np.unique(f.index):
    s=np.array(f.loc[i])
    if s.ndim==1:
        ss.append([s.tolist()])

    else:
        ss.append(s.tolist())
#seq=np.array(ss)
#print(seq.ndim)

seq=sequence.pad_sequences(np.array(ss),maxlen=6,dtype='float',padding='post')
#print(seq)


# define input sequence
#sequence = array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],[0.5, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])

#%%
# reshape input into [samples, timesteps, features]
n_sample=seq.shape[0]
n_feature = seq.shape[2]
n_timestep = seq.shape[1]


#seq=seq.reshape(cn,12,n_feature)
#seq=seq.reshape(seq.shape[0],1,seq.shape[1])
# define model
#%%
model = Sequential()
model.add(Masking(mask_value=0.0,input_shape=(n_timestep,n_feature)))
#model.add(LSTM(100, activation='relu', input_dim=n_feature))
#model.add(LSTM(100, activation='relu', input_shape=( None,n_feature)))

#%%
#model.add(RepeatVector(1))
model.add(LSTM(n_feature, activation='relu', return_sequences=False))#False?
#model.add(TimeDistributed(Dense(n_feature)))

#sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='mean_squared_error', optimizer=sgd)
model.compile(optimizer='adam', loss='mse')
# fit model
#model.fit(seq, seq, epochs=50, verbose=1)
plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# demonstrate recreation
yhat = model.predict(seq, verbose=1)
#print(yhat[0,:,0])
print(yhat)
#print(len(yhat))
#print(len(yhat[0]))
#print(len(yhat[0][0]))

pd.DataFrame(yhat).to_csv('lstmdecode5.csv')