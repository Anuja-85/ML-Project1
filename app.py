from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st


start = '2013-08-31'
end = '2023-07-30'


st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)
df.head()

# Describe Data
st.subheader('Data from 2013 - 2023')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


# Spliting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

close = df['Close']
X = close.values
X_shaped=X.reshape(-1,1)

scalar = MinMaxScaler(feature_range=(0, 1))
X_scaled=scalar.fit_transform(X_shaped)


# Load Model
model = load_model('keras_model.h5')

# Testing Part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scalar.fit_transform(final_df)

thresh = int(len(X)*.7)
test_data = X_scaled[thresh:,:]
xtest=[]
ytest=[]
for i in range(100,len(test_data)):
    xtest.append(input_data[i-100:i,0])
    ytest.append(input_data[i,0])
xtest= np.array(xtest)
ytest=np.array(ytest)
xtest= np.reshape(xtest,(xtest.shape[0],xtest.shape[1],1))

ypred = model.predict(xtest)
scalar = scalar.scale_

scale_factor = 1/scalar[0]
ypred = ypred * scale_factor
ytest = ytest * scale_factor

# Final Graph
st.subheader('Prediction vs Original')
fig2=plt.figure(figsize=(16, 8))
plt.plot(ypred, color='red', label='Predicted Price')
plt.plot(ytest, color='blue', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
