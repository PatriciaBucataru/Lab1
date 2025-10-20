
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve, auc


X_train, X_test, y_train, y_test = generate_data(
    n_train=400,
    n_test=100,
    n_features=2,
    contamination=0.1,
    random_state=42
)


model = KNN(contamination=0.1) 
model.fit(X_train)


y_train_pred = model.predict(X_train) 
y_test_pred = model.predict(X_test)


cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix:")
print(cm)
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


ba = balanced_accuracy_score(y_test, y_test_pred)
print(f"Balanced Accuracy: {ba:.4f}")


y_test_scores = model.decision_function(X_test)  
fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve - KNN')
plt.legend()
plt.show()