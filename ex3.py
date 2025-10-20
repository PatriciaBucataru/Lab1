import numpy as np
from pyod.utils.data import generate_data
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


X_train, _, y_train, _ = generate_data(
    n_train=1000,
    n_test=0,
    n_features=1,
    contamination=0.1,
    random_state=42
)
X_train = X_train.ravel()


mean = np.mean(X_train)
std = np.std(X_train)
z_scores = (X_train - mean) / std


contamination = 0.1
threshold = np.quantile(np.abs(z_scores), 1 - contamination)


y_pred = (np.abs(z_scores) >= threshold).astype(int)


cm = confusion_matrix(y_train, y_pred)
tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix:")
print(cm)
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

ba = balanced_accuracy_score(y_train, y_pred)
print(f"Balanced Accuracy: {ba:.4f}")
