import matplotlib.pyplot as plt
from pyod.utils.data import generate_data


X_train, X_test, y_train, y_test = generate_data(
    n_train=400,
    n_test=100,
    n_features=2,
    contamination=0.1,
    random_state=42
)


plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], label='Inliers', alpha=0.7)
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], color='red', marker='x', label='Outliers')
plt.legend()
plt.show()
