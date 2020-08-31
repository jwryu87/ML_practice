import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

# 데이터 적재
oecd_bli = pd.read_csv(r"c:\Users\jwryu87\OneDrive\work\Python\ML_practice\DATA\oecd_bli_20185.csv", thousands=',')
gdp_per_capita = pd.read_csv(r"c:\Users\jwryu87\OneDrive\work\Python\ML_practice\DATA\gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")

# 데이터 준비
country_status = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
Y = np.c_[country_stats["Life satisfaction"]]

# 데이터 시각화
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

# 선형 모델 선택
model = sklearn.linear_model.LinearRegression()

# 모델 훈련
model.fit(X, y)

# 키프로스에 대한 예측
X_new = [[22587]] # 키프로스 1인당 GDP
print(model.predict(X_new)) # 결과 [[5.96242338]]

