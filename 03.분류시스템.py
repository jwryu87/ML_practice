import numpy as np
# MNIST 데이터셋
# 새로운 분류 알고리즘이 나올때마다 이 데이터셋에서 얼마나 잘 작동하는지 보자.

# from sklearn.datasets import fetch_openml # scikit-learn 0.20 이후 부터는 fetch_mldata( )는 더이상 지원하지 않는 것을 알 수 있었습니다. (fetch_mldata -> fetch_openml) 로 교체
# mnist = fetch_openml('mnist_784') # mnist = fetch_mldata('MNIST original')
# print(mnist.data.shape)
# print(mnist.target.shape)

# 왜인지 모르겠지만 위의 소스로 다운이 안되서 아래 소스로 다운함
from sklearn.datasets  import load_digits
mnist = load_digits()
X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)

# def sort_by_target(mnist):
#     reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
#     reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
#     mnist.data[:60000] = mnist.data[reorder_train]
#     mnist.target[:60000] = mnist.target[reorder_train]
#     mnist.data[60000:] = mnist.data[reorder_test + 60000]
#     mnist.target[60000:] = mnist.target[reorder_test + 60000]
#
# try:
#     from sklearn.datasets import fetch_openml
#     mnist = fetch_openml('mnist_784', version=1, cache=True)
#     mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
#     sort_by_target(mnist) # fetch_openml() returns an unsorted dataset
# except ImportError:
#     from sklearn.datasets import fetch_mldata
#     mnist = fetch_mldata('MNIST original')

import matplotlib
import matplotlib.pyplot as plt

# 출력해보기
# some_digit = X[360]
# some_digit_image = some_digit.reshape(8, 8)
# plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()

X_train, X_test, y_train, y_test = X[1500:], X[1500:], y[1500:], y[1500:]

import numpy as np
shuffle_index = np.random.permutation(1500)
# X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# print(X_train, y_train)


########################################################################################################################
# 이준 분류기 훈련

y_train_5 = (y_train == 5) # 5는 True고 다른 숫자는 모두 False
y_test_5 = (y_test == 5)

# 사이킷런의 SGDClassifier 클래스를 사용해 확률적(=무작위성) 경사 하갈법(SGD) 분류기로 시작해보는것도 나쁘지 않습니다.
# 이 분류기는 매우 큰 데이터셋을 효율적으로 처리하는 장점을 지니고 있습니다.
# SGD가 한번에 하나씩 훈련 샘플을 독립적으로 처리하기 때문입니다.
# 그래서 SGD가 온라인 학습에 잘 들어맞습니다.
from sklearn.linear_model import  SGDClassifier

sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train_5)


#**** 애초에 데이터 로딩부터가 잘 안되니 뒤에 아무것도 못하겠네