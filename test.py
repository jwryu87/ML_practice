import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

print(HOUSING_PATH)
print(HOUSING_URL)

housing_url = HOUSING_URL
housing_path = HOUSING_PATH

print(os.path.isdir(housing_path))

if not os.path.isdir(housing_path):
    os.makedirs(housing_path)
tgz_path = os.path.join(housing_path, "housing.tgz")

print(tgz_path)

urllib.request.urlretrieve(housing_url, tgz_path)
housing_tgz = tarfile.open(tgz_path)
housing_tgz.extractall(path=housing_path)
housing_tgz.close()



import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return print(pd.read_csv(csv_path))

load_housing_data(housing_path=HOUSING_PATH)