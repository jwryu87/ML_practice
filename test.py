from sklearn.datasets  import load_digits
mnist = load_digits()
X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)