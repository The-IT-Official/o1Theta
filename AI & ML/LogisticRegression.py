from sklearn.linear_model import LogisticRegression
import numpy as np

# Hours studied (X)
X = np.array([[1], [2], [3], [4], [5], [6]])
# Pass (1) or Fail (0)
y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

prediction = model.predict([[3.5]])
probability = model.predict_proba([[3.5]])

print("Pass (1) or Fail (0)?", prediction[0])
print("Probability of Passing:", probability[0])