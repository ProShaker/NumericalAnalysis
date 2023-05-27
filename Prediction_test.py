import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# 주식 종목 및 기간 설정
company = 'PFE'
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 5, 25)

# 주식 데이터 가져오기
data = yf.download(company, start=start_date, end=end_date)

# 데이터 전처리를 위한 스케일러 설정
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# 예측에 사용할 일 수 설정
prediction_days = 60

# x_train, y_train 데이터 생성
x_train = []
y_train = []
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Sequential 모델 생성
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# 테스트 데이터 로드
test_start = end_date + dt.timedelta(days=1)
test_end = end_date + dt.timedelta(days=1) + dt.timedelta(days=prediction_days)
test_data = yf.download(company, start=test_start, end=test_end)
actual_prices = test_data['Close'].values

# 전체 데이터셋 생성
total_dataset = pd.concat((data['Close'], test_data['Close']))
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# 데이터에 대한 예측 수행
# 테스트 데이터의 예측에 사용할 입력 데이터인 x_test를 생성합니다.
x_test = []

# 현재 일수로부터 prediction_days 일 이전까지의 데이터를 x_test에 추가합니다.
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
x_test = np.expand_dims(x_test, axis=2)

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
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1]))
real_data = np.expand_dims(real_data, axis=2)

# 모델을 사용하여 다음 날의 예측 수행
prediction = model.predict(real_data)
# 스케일링된 예측 결과를 원래 값으로 되돌립니다.
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

# 예측 데이터와 실제 데이터 비교를 위해 스케일링된 데이터를 되돌립니다.
predicted_prices = scaler.inverse_transform(predicted_prices).flatten()
actual_prices = test_data['Close'].values

# 예측값과 실제값을 기준으로 이진 분류를 수행합니다.
# 상승한 경우를 1, 하락한 경우를 0으로 레이블링합니다.
predicted_labels = np.where(predicted_prices > actual_prices, 1, 0)
actual_labels = np.where(actual_prices > actual_prices.shift(-1), 1, 0)
actual_labels = actual_labels[:-1]  # 마지막 값 제거 (다음 날 예측이 없으므로)

# 혼동 행렬 생성
cm = confusion_matrix(actual_labels, predicted_labels)
accuracy = np.trace(cm) / np.sum(cm)

# 혼동 행렬 시각화
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt=".0f", cmap='Blues', square=True)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
plt.show()