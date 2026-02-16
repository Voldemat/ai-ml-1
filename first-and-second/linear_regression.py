import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(1)

x = rng.integers(low=1, high=1000, size=10).reshape(-1, 1)
y = rng.integers(low=1, high=1000, size=10).reshape(-1, 1)

model = LinearRegression().fit(x, y)

print("Slope (price per m2): ", model.coef_[0])
print("Base price: ", model.intercept_)

new_area = 90
predicted_price = model.predict([[new_area]])

print("Predicted price for", new_area, "m²:", predicted_price[0])
