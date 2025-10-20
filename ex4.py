import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


np.random.seed(42)

n_samples = 1000
n_features = 2       
contamination = 0.1 


mu = np.array([2.0, -1.0])
Sigma = np.array([[1.0, 0.5],
                  [0.5, 1.5]])

X = np.random.randn(n_samples, n_features)
L = np.linalg.cholesky(Sigma)
Y = X @ L.T + mu


y_true = np.zeros(n_samples, dtype=int)
n_outliers = int(contamination * n_samples)
outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
y_true[outlier_indices] = 1


Y[outlier_indices] += np.array([5.0, 5.0]) 


Sigma_inv = np.linalg.inv(Sigma)
z_scores = np.array([np.sqrt((y - mu) @ Sigma_inv @ (y - mu)) for y in Y])


threshold = np.quantile(z_scores, 1 - contamination)
y_pred = (z_scores >= threshold).astype(int)


cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix:")
print(cm)
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

ba = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced Accuracy: {ba:.4f}")

plt.figure(figsize=(6,6))
plt.scatter(Y[y_pred==0,0], Y[y_pred==0,1], s=20, c='green', label='Normal')
plt.scatter(Y[y_pred==1,0], Y[y_pred==1,1], s=20, c='red', label='Detected Anomaly', marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Multivariate Anomaly Detection (Balanced Acc = {ba:.2f})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
