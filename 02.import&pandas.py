import os
import tarfile
from six.moves import urllib
import pandas as pd

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# 데이터 import
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

# csv 파일을 pandas로 읽기
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path) # pandas 로 데이터를 올리는 메서드

if __name__ == "__main__":
    fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
    housing = load_housing_data(housing_path=HOUSING_PATH)

    # pandas 연습
    # print(housing.head())
    # print(housing.info()) # info는 데이터에 대한 간략한 설명을 보여줌(전체 행 수, 각 특성의 데이터 타입과 null이 아닌 값의 개수)
    # print(housing["ocean_proximity"].value_counts()) # value_counts 메서드
    # print(housing.describe()) # 숫자형 특성의 요약정보를 알려줌
    ###
    import matplotlib.pyplot as plt # 데이터 형태를 빠르게 검토하는 다른 방법은 각 숫자형 특성을 히스토그램으로 그려보는 것입니다.
    housing.hist(bins=50, figsize=(20, 15))
    plt.show()
    ###


