import numpy as np
from sklearn.linear_model import LogisticRegression

data = np.array([
    [80, 20],
    [90, 18],
    [60, 25],
    [30, 30],
    [85, 19]
])

labels = np.array([1, 1, 0, 0, 1])


model = LogisticRegression().fit(data, labels)

prediction = model.predict([[60, 22]])
print("Will it rain?", "Yes" if prediction[0] == 1 else "No")
