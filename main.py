import numpy as np
from sklearn.linear_model import LinearRegression

x = np.random.randint(low=1, high=1000, size=10).reshape(-1, 1)
y = np.random.randint(low=1, high=1000, size=10)

model = LinearRegression().fit(x, y)

print("Slope (price per m2): ", model.coef_[0])
print("Base price: ", model.intercept_)
