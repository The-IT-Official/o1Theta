from sklearn.linear_model import LinearRegression
import numpy as np

# Years of Experience (X)
X = np.array([[1], [2], [3], [4], [5]])
# Salaries (y)
y = np.array([40000, 45000, 50000, 55000, 60000])

model = LinearRegression()
model.fit(X, y)
prediction = model.predict([[7]])
print("Predicted Salary:", prediction[0])