import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
ticker = input("기업 별 ticker를 입력해주세요. : ")
company = {
    'PFE': 'Pfizer(PFE)',
    'CVS': 'CVS Health Corp. (CVS)',
    'UNH': 'UnitedHealth Group Inc. (UNH)',
    'MCK': 'McKesson Corp. (MCK)',
    'ABC': 'AmerisourceBergen Corp. (ABC)',
    'CAH': 'Cardinal Health Inc. (CAH)',
    'CI': 'Cigna Corp. (CI)',
    'ELV': 'Elevance Health (ELV)',
    'CNC': 'Centene Corp. (CNC)',
    'WBA': 'Walgreens Boots Alliance Inc. (WBA)'
}
# CSV 파일 읽기

data = pd.read_csv("D:\GITHUB/NumericalAnalysis\Top 10 Healthcare Companies in the United States/" + company[ticker.upper()] + ".csv", encoding='cp949')


# 필요한 열 선택
df = data[['Date', 'Close', 'Volume', 'PriceRange']]

# 날짜를 datetime 형식으로 변환
df['Date'] = pd.to_datetime(df['Date'])

# 2000.01.01부터 이번주까지의 데이터 추출
start_date = pd.to_datetime('2000-01-01')
end_date = pd.to_datetime('2023-05-10')
mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
df = df.loc[mask]

# 독립변수와 종속변수 설정
X = df[['Volume', 'PriceRange']]
y = df['Close']

# 다항식 변환
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 모델 학습
model = LinearRegression()
model.fit(X_poly, y)

# 예측값 계산
y_pred = model.predict(X_poly)

# 그래프 그리기
plt.plot(df['Date'], y, label='Real')
plt.plot(df['Date'], y_pred, label='Prediction')
plt.xlabel('Date')
plt.ylabel('Close')


plt.title('Polynomial Regression')
plt.legend()
plt.xticks(rotation=45)
plt.show()



