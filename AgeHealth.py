import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "NanumGothic"


# CSV 파일 읽기
data = pd.read_csv("C:/Users/김형준/Documents/GitHub/NumericalAnalysis/헬스장설문조사.csv")

# 연령대별 그룹화
groups = data.groupby('Age')

# 운동목적과 총 인원 수 구하기
purposes = data['Purpose'].unique()
num_purposes = len(purposes)
# 여성 그래프 그리기
female_data = data[data['Gender'] == '여']
female_groups = female_data.groupby('Age')
for i, (age, group) in enumerate(female_groups):
    purpose_counts = group.groupby('Purpose').size()
    purpose_counts = purpose_counts.reindex(purposes, fill_value=0)
    plt.bar(range(num_purposes), purpose_counts,
            label=age, alpha=0.5, color='pink')

# 남성 그래프 그리기
male_data = data[data['Gender'] == '남']
male_groups = male_data.groupby('Age')
for i, (age, group) in enumerate(male_groups):
    purpose_counts = group.groupby('Purpose').size()
    purpose_counts = purpose_counts.reindex(purposes, fill_value=0)
    plt.bar(range(num_purposes), purpose_counts,
            label=age, alpha=0.5, color='blue')

# 그래프 레이블 및 제목 설정
plt.xlabel('Exercise Purpose')
plt.ylabel('Count')
plt.title('Exercise Purpose by Age Group')
plt.xticks(range(num_purposes), purposes, rotation=45)  # 라벨의 회전값 조정
plt.legend(title='Age')
plt.tight_layout()

# 그래프 출력
plt.show()