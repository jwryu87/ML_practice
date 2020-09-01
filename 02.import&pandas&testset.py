import os
import tarfile
from six.moves import urllib
import pandas as pd

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

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


