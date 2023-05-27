# ‪C:\Users\dktk5\Desktop\3학년_1학기\교과\공수해\Top 10 Healthcare Companies in the United States.xlsx
# https://youtu.be/4RE5a98rI-M

'''------------------------------------------------------

read data
------------------------------------------------------'''

import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\dktk5\Desktop\3학년_1학기\교과\공수해\Top 10 Healthcare Companies in the United States\Pfizer(PFE).csv", encoding='cp949')

df = df.dropna()

x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

print(x)
print(y)
'''------------------------------------------------------
Preprocessing

train test split
------------------------------------------------------'''

from sklearn.model_selection import train_test_split
x_main, x_test, y_main, y_test = train_test_split(x, y, test_size=0.75, stratify=y)
# x_train, x_validation, y_train, y_validation = train_test_split(x_main, y_main, test_size=0.25, stratify=y_main)

# scaling
print(y.shape)
print(x.shape)