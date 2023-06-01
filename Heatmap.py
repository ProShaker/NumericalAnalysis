import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\dktk5\Desktop\3학년_1학기\교과\공수해\Top 10 Healthcare Companies in the United States\Pfizer(PFE).csv", encoding='cp949')

# 독립변수와 종속변수 설정
X = data[['Volume', 'PriceRange']]


# 상관 행렬 계산
corr_matrix = X.corr()

# Heatmap 그리기
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Heatmap of Correlation: 거래량 vs 가격변동폭')
plt.show()