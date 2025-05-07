from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([[1], [2], [3], [4]])
y = np.array([0, 0, 0, 0])

model = LogisticRegression()
model.fit
# print(x + y) # element-wise addition
# print(x * y) # element-wise multiplication
# print(x @ y) # matrix multiplication
# print(torch.matmul(x, y)) # same
# print(x.T) # transpose
# print(x.shape)
# print(x[0])
# print(x[:, 1])