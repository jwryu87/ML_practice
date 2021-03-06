import os
import tarfile
from six.moves import urllib
import pandas as pd
import sys


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# 가. 문제 정의
# 나. 데이터 로딩 후 탐색
# 다. 훈련세트, 테스트세트
# 라. 변환 파이프라인 (ml 알고리즘에 주입할 데이터를 자동으로 정제하고 준비)

########################################################################################################################
# (1) 데이터 import
# - url로 접속하여 압축파일 다운로드 후
# - 디렉토리 만들고 압축을 풀어 csv 파일로 만들기
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()



########################################################################################################################
# (2) pandas 연습
# csv 파일을 pandas로 읽기
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path) # pandas 로 데이터를 올리는 메서드

if __name__ == "__main__":
    fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
    housing = load_housing_data(housing_path=HOUSING_PATH)

    # print(housing.head())
    # print(housing.info()) # info는 데이터에 대한 간략한 설명을 보여줌(전체 행 수, 각 특성의 데이터 타입과 null이 아닌 값의 개수)
    # print(housing["ocean_proximity"].value_counts()) # value_counts 메서드
    # print(housing.describe()) # 숫자형 특성의 요약정보를 알려줌
    ###
    # import matplotlib.pyplot as plt # 데이터 형태를 빠르게 검토하는 다른 방법은 각 숫자형 특성을 히스토그램으로 그려보는 것입니다.
    # housing.hist(bins=50, figsize=(20, 15))
    # plt.show()
    ###


########################################################################################################################
# (3) test data 만들기
# 3-1 하지만 이것은 다시 실행하면 다른 테스트 세트가 되어 버린다.
import numpy as np

# def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]
#
# train_set, test_set = split_train_test(housing, 0.2)
# print(len(train_set), "train +", len(test_set), "test")

# 3-2 해결책으로 샘플의 식별자를 사용하여 테스트 세트로 보낼지 말지를 정하는것
# - 예를 들어 각 샘플마다 식별잘의 해시값을 계산하여 해시의 마지막 바이트의 값이 51(256의 20% 정도)보다 작거나 같은 샘플만 테스트 세트로 보낼 수 있습니다.

from  zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_:test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# 주택 데이터 셋에는 식별자 컬럼이 없으므로 대신 행의 인덱스를 ID로 사용
# housing_with_id = housing.reset_index() # index 열이 추가된 데이터 프레임을 반환합니다.
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# 3-3 사이킷런은 데이터셋을 여러 서브셋으로 나누는 다양한 방법을 제공합니다.
# - train_test_split
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index] # 훈련세트
    strat_test_set = housing.loc[test_index]   # 테스트세트

# 테스트 세트 생성은 머신러닝 프로젝트에서 아주 중요한 부분이다.



########################################################################################################################
# 데이터 이해를 위한 탐색과 시각화

# (1) 시각화

import matplotlib.pyplot as plt

housing = strat_train_set.copy()

# 산점도 그리기
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show() # 출력 시 이게 꼭 필요함

# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#              s=housing["population"]/100, label="population", figsize=(10,7),
#              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False
#              )
# plt.legend()
# plt.show()
# -> 주택가격은 지역과 인구밀도에 관련이 매우 크다는 사실


# (2) 상관관계 조사
# corr_matrix = housing.corr()
# corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix # 상관관계를 확인하는 다른 방법으로 숫자형 특성 사이에 산점도를 그려주 pandas 함수 사용
# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show() # 역시 이거 있어야 함

# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1) # 가장 상관관계가 높은 것과 산점도 그리기
# plt.show()


# (3) 특성 조합으로 실험
# housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# housing["population_per_household"] = housing["population"]/housing["households"]
#
# corr_matrix = housing.corr()
# corr_matrix["median_house_value"].sort_values(ascending=False)


########################################################################################################################
# 머신러닝 알고리즘을 위한 데이터 준비
# - 함수를 만들어서 자동으로 처리되도록 해야 함

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# (1) 데이터 정제
# 1-1 숫자형
#  - total_bedrooms 특성에 값이 없는 경우를 보았는데 이를 고쳐보겠습니다.

# # * 해당 구역을 제거합니다
# housing.dropna(subset=["total_bedrooms"])
# # * 전체 특성을 삭제합니다
# housing.drop("total_bedrooms", axis=1)
# # * 어떤 값으로 채웁니다 (0, 평균, 중간값)
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)

# 사이킷런에서 Imputer 는 누락된 값을 손쉽게 다루도록 해줍니다.
# --> Imputer 는 사용할 수가 없는데 사이킷런 0.20 버전에서 사용 중지 경고가 발생했다고 함. 아마 없어진듯. 다른것으로 대체해야 할거 같다
# from sklearn.preprocessing import Imputer
# imputer = Imputer(strategy="median")
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

# 중간값이 수치형 특성에서만 계산될 수 있기 때문에 텍스트 특성인 컬럼을 (ocean_proxomity) 제외한 데이터 복사본을 생성한다.
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num) # 학습된 imputer 객체를 사용해 훈련세트에서 누락된 값을 학습한 중간값으로 바꿀 수 있다.
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=list(housing.index.values)) # 이 결과는 변형된 특성들이 들어 있는 평범한 넘파이 배열입니다. 이를 다시 판다스 데이터플임으로 간단히 되돌릴 수 있습니다.

# 1-2 텍스트와 범주형
# 대부분의 머신러닝 알고리즘은 숫자형을 다루므로 이 카테고리를 텍스트에서 숫자로 바꾸도록 함
# 각 카테고리를 다른 정수값으로 매핑해주는 판다스의 factorize() 메서드를 사용함
housing_cat = housing["ocean_proximity"]
housing_cat.head(10)
housing_cat_encoded, housing_categories = housing_cat.factorize()
housing_cat_encoded[:10]
housing_categories # 카테고리도 보여줌
# print(housing_cat_encoded)

# 원-핫 인코딩: 카테고리별 이진특성으로 변경
# 사이킷런은 숫자로 된 범주형 값을 원-핫 벡터로 바꿔주는 OneHotEncoder 를 제공
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
housing_cat_1hot.toarray()

# 위처럼 텍스트 카테고리를 숫자 카테고리로, 숫자 카테고리를 원-핫 벡터로 바꿔주는 이 두 가지 변환을 CategoricalEncoder 를 사용하여 한번에 처리할 수 있다 ;;
# 하지만 CategoricalEncoder 이것도 왜 인지 모르겠지만 불러올 수가 없어서 넘어감
from sklearn.preprocessing import OrdinalEncoder # -> 제작자가 만들어놓고 배포하지 않아서 실행이 안된다고 한다. 황당하다. 그래서 (CategoricalEncoder -> OrdinalEncoder) 로 바꿈
cat_encoder = OrdinalEncoder()
# housing_cat_reshaped = housing_cat.value.reshape(-1, 1)
# housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)


# 1-3 나만의 변환기
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_rooms = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # 더 할 일이 없습니다
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_rooms:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)



########################################################################################################################
# 변환 파이프라인
# 사이킷런에는 연속된 변환을 순서대로 처리할 수 있도록 도와주는 Pipeline 클래스가 존재함

# (1)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('str_scaler', StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

# (2)
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

# (3)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', OrdinalEncoder()),
])

# (4)

from sklearn.pipeline import FeatureUnion # 두 파이프라인을 하나로 합치기

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

# (5)
housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared.shape





########################################################################################################################
# 모델 선택과 훈련
from sklearn.linear_model import LinearRegression # 선형회귀

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# 이게 끝입니다!

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
# print("예측: ", lin_reg.predict(some_data_prepared))
# print("레이블: ", list(some_labels))

# 이 회귀모델의 RMSE를 측정해보겠습니다
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)

# 다른 모델: DecisionTreeRegressor 을 사용한 비선형 어쩌고
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
# print(tree_rmse)



########################################################################################################################
# 교차검증을 사용한 평가
# 사이킷런의 교차 검증 기능을 사용: K-겹 교차검증
# 훈련 세트를 폴드라 불리는 10개의 서브셋으로 무작위로 분할 후 결정트리 모델을 10번 훈현하고 평가

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# display_scores(tree_rmse_scores) # 결정트리 모델

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
# display_scores(lin_rmse_scores) # 선형회귀 모델


# from sklearn.ensemble import RandomForestRegressor
# forest_reg = RandomForestRegressor()
# forest_reg.fit(housing_prepared, housing_labels)
# forest_rmse
# display_scores(forest_rmse_scores) # 랜덤포레스트 모델 (이 부분의 소스는 샘플이어서 실행 시킬 수 없다)




















