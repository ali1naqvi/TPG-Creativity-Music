# %%
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# %%
df = pd.read_csv('./input.csv', nrows=950)

# %%
df.shape

# %%
df.head()

# %%
df.tail()

# %%
test_split=round(len(df)*0.20)

# %%
test_split

# %%
df_for_training=df[:-1041]
df_for_testing=df[-1041:]

# %%
print(df_for_training.shape)
print(df_for_testing.shape)

# %%
scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)

# %%
df_for_testing_scaled=scaler.transform(df_for_testing)

# %%
df_for_training_scaled

# %%
df_for_training_scaled.shape

# %%
df_for_testing_scaled.shape

# %%
def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)        

# %%
trainX,trainY=createXY(df_for_training_scaled,30)

# %%
trainX.shape

# %%


# %%
testX,testY=createXY(df_for_testing_scaled,30)

# %%
trainX[0]

# %%
print("trainX Shape-- ",trainX.shape)
print("trainY Shape-- ",trainY.shape)

# %%
print("testX Shape-- ",testX.shape)
print("testY Shape-- ",testY.shape)

# %%
print("trainX[0]-- \n",trainX[0])
print("\ntrainY[0]-- ",trainY[0])

# %%
trainY[0]

# %%
trainY.shape

# %%
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

# %%
def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(50,return_sequences=True,input_shape=(30,5)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model

grid_model = KerasRegressor(build_fn=build_model,verbose=1,validation_data=(testX,testY))
parameters = {'batch_size' : [16,20],
              'epochs' : [8,10],
              'optimizer' : ['adam','Adadelta'] }

grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)

# %%
grid_search = grid_search.fit(trainX,trainY)

# %%
grid_search.best_params_

# %%
my_model=grid_search.best_estimator_.model

# %%
my_model

# %%
prediction=my_model.predict(testX)

# %%
print("prediction\n", prediction)
print("\nPrediction Shape-",prediction.shape)

# %%
prediction.shape

# %%
scaler.inverse_transform(prediction)

# %%
prediction_copies_array = np.repeat(prediction,5, axis=-1)

# %%
prediction_copies_array.shape

# %%
prediction_copies_array

# %%
pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),5)))[:,0]

# %%
pred

# %%
original_copies_array = np.repeat(testY,5, axis=-1)

original_copies_array.shape

original=scaler.inverse_transform(np.reshape(original_copies_array,(len(testY),5)))[:,0]

# %%
pred

# %%
print("Pred Values-- " ,pred)
print("\nOriginal Values-- ",original)

# %%


# %%
import matplotlib.pyplot as plt

# %%
plt.plot(original, color = 'red', label = 'Real  Stock Price')
plt.plot(pred, color = 'blue', label = 'Predicted  Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()

# %%


# %%
df_30_days_past=df.iloc[-30:,:]

# %%
df_30_days_past

# %%
df_30_days_future=pd.read_csv("test.csv",parse_dates=["Date"],index_col=[0])
df_30_days_future.shape

# %%
df_30_days_future

# %%
df_30_days_future["Open"]=0
df_30_days_future=df_30_days_future[["Open","High","Low","Close","Adj Close"]]
old_scaled_array=scaler.transform(df_30_days_past)
new_scaled_array=scaler.transform(df_30_days_future)
new_scaled_df=pd.DataFrame(new_scaled_array)
new_scaled_df.iloc[:,0]=np.nan
full_df=pd.concat([pd.DataFrame(old_scaled_array),new_scaled_df]).reset_index().drop(["index"],axis=1)

# %%
full_df.shape

# %%
full_df.tail()

# %%
full_df.shape

# %%
full_df_scaled_array=full_df.values

# %%
full_df_scaled_array.shape

# %%
all_data=[]
time_step=30
for i in range(time_step,len(full_df_scaled_array)):
    data_x=[]
    data_x.append(full_df_scaled_array[i-time_step:i,0:full_df_scaled_array.shape[1]])
    data_x=np.array(data_x)
    prediction=my_model.predict(data_x)
    all_data.append(prediction)
    full_df.iloc[i,0]=prediction

# %%
all_data

# %%
new_array=np.array(all_data)
new_array=new_array.reshape(-1,1)
prediction_copies_array = np.repeat(new_array,5, axis=-1)
y_pred_future_30_days = scaler.inverse_transform(np.reshape(prediction_copies_array,(len(new_array),5)))[:,0]

# %%
y_pred_future_30_days

# %%


# %%
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

# %%
mymodel.save('Model_future_value.h5')
print('Model Saved!')

# %%
scaler

# %%
import pickle
scalerfile = 'scaler_model_future_value.pkl'
pickle.dump(scaler, open(scalerfile, 'wb'))

# %% [markdown]
# # END!!!!


