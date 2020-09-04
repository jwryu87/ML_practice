import numpy as np
# MNIST 데이터셋
# 새로운 분류 알고리즘이 나올때마다 이 데이터셋에서 얼마나 잘 작동하는지 보자.

from sklearn.datasets import fetch_openml # scikit-learn 0.20 이후 부터는 fetch_mldata( )는 더이상 지원하지 않는 것을 알 수 있었습니다. (fetch_mldata -> fetch_openml) 로 교체
mnist = fetch_openml('mnist_784', version=1) # mnist = fetch_mldata('MNIST original')
print(mnist)

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
