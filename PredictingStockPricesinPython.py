# https://youtu.be/PuZY9q-aKLw
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as pdr
import datetime as dt
import yfinance as yfin

# yfinance로부터 데이터를 가져올 수 있도록 설정
yfin.pdr_override()

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


'''------------------------------------------------------------------

필요한 라이브러리 및 모듈을 임포트합니다.
주식 데이터를 가져올 회사를 선택하고, 데이터의 시작일과 종료일을 설정합니다.
데이터를 가져옵니다.
데이터 전처리를 위해 스케일러를 설정하고, 종가 데이터를 스케일링합니다.
예측에 사용할 일 수를 설정합니다.
훈련 데이터인 x_train과 y_train을 생성합니다.
훈련 데이터를 numpy 배열로 변환하고 LSTM 모델에 입력하기 위해 차원을 재구성합니다.
LSTM 모델을 구성하고, 컴파일합니다.
훈련 데이터를 사용하여 모델을 훈련합니다.
테스트 데이터를 설정하고, 실제 주가를 가져옵니다.
전체 데이터에 대한 예측 결과를 얻기 위해 훈련 데이터와 테스트 데이터를 연결합니다.
테스트 데이터에 대한 모델 입력을 준비합니다.
모델을 사용하여 테스트 데이터에 대한 예측을 수행합니다.
예측 결과를 스케일링된 값에서 원래 값으로 변환합니다.
실제 주가와 예측 주가를 그래프로 그려 비교합니다.
다음 날의 예측을 수행하기 위해 모델 입력을 준비합니다.
모델을 사용하여 다음 날의 예측을 수행합니다.
예측 결과를 출력합니다.

------------------------------------------------------------------'''

# 회사 이름 설정 (예: AAPL은 애플 주식)
company = 'PFE'

start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 5, 25)

# 주식 데이터 가져오기
data = pdr.get_data_yahoo(company, start_date, end_date)

# 데이터 전처리를 위한 스케일러 설정
# 데이터 전처리를 위해 MinMaxScaler를 사용하여 데이터를 0과 1 사이의 범위로 스케일링합니다.
# 'Close' 컬럼의 값을 2차원 배열로 변환하고, scaler를 사용하여 스케일링합니다.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
print(len(scaled_data))

# 예측에 사용할 일 수 설정
# 이는 모델이 과거 데이터를 기반으로 다음 종가를 예측할 때 참조할 일 수입니다.
prediction_days = 60

# x_train은 예측을 위한 입력 데이터로 사용되고, y_train은 예측 대상인 다음 종가 데이터입니다.
x_train = []
y_train = []

# x_train, y_train 데이터 생성
# x_train과 y_train 데이터를 생성하기 위해 반복문을 사용합니다.
# x_train은 예측에 사용될 과거 60일의 데이터를 담고 있고, y_train은 그 다음 날의 종가 값을 담고 있습니다.
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])


# x_train과 y_train을 numpy 배열로 변환합니다.
# LSTM 모델에 입력하기 위해 x_train의 차원을 재구성합니다. (샘플 개수, 타임스텝, 특성) 형태로 변환합니다.
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Sequential 모델을 생성합니다.
model = Sequential()

'''
다음 종가 값을 예측하는 LSTM 레이어 추가: LSTM 레이어를 모델에 추가하여 다음 종가 값을 예측합니다.
LSTM 레이어를 차례로 추가
모델을 컴파일하고 주어진 횟수만큼 학습합니다.
'''
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# Dropout을 사용하여 과적합을 방지
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# 다음 종가 값을 예측하는 LSTM 레이어 추가
model.add(LSTM(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

''' Test The Model Accuracy on Existing Data'''

# 테스트 데이터 로드
test_start = start_date
test_end = dt.datetime.today()

test_data = pdr.get_data_yahoo(company, test_start, test_end)
actual_prices = test_data['Close'].values

#total_dataset: 학습 데이터와 테스트 데이터를 연결하여 전체 데이터셋을 만듭니다.'''
total_dataset = pd.concat((data['Close'], test_data['Close']))

# model_inputs: 모델 입력 데이터를 생성하기 위해 테스트 데이터에서 예측에 사용할 기간에 해당하는 데이터를 가져옵니다.
# 이 데이터는 스케일링되고 모델에 입력될 준비가 됩니다.
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

# 데이터에 대한 예측 수행
# 테스트 데이터의 예측에 사용할 입력 데이터인 x_test를 생성합니다.
x_test = []

# 현재 일수로부터 prediction_days 일 이전까지의 데이터를 x_test에 추가합니다.
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 모델을 사용하여 x_test에 대한 예측 수행
predicted_prices = model.predict(x_test)
# 스케일링된 예측 결과를 원래 값으로 되돌립니다.
predicted_prices = scaler.inverse_transform(predicted_prices)

# 테스트 예측 결과를 그래프로 그리기
plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
plt.plot(predicted_prices, color="red", label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()

# 다음 날의 예측 수행
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs)]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

# 모델을 사용하여 다음 날의 예측 수행
prediction = model.predict(real_data)
# 스케일링된 예측 결과를 원래 값으로 되돌립니다.
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

acc = accuracy_score(x_train, predicted_prices)