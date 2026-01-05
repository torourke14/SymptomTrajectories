import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
from .xgboost import precision_recall_curve

def plot_pr_curve(y_true, y_prob, title, out_path):
  p, r, _ = precision_recall_curve(y_true, y_prob)
  plt.figure()
  plt.plot(r, p)
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.title(title)
  plt.grid(True, linestyle=":", linewidth=0.5)
  plt.tight_layout()
  plt.savefig(out_path, dpi=150)
  plt.close()

def plot_roc_curve(y_true, y_prob, title, out_path):
  fpr, tpr, _ = roc_curve(y_true, y_prob)
  plt.figure()
  plt.plot(fpr, tpr)
  plt.plot([0,1],[0,1], linestyle="--")
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title(title)
  plt.grid(True, linestyle=":", linewidth=0.5)
  plt.tight_layout()
  plt.savefig(out_path, dpi=150)
  plt.close()

def plot_confusion(y_true, y_prob, threshold, title, out_path):
  y_pred = (y_prob >= threshold).astype(int)
  cm = confusion_matrix(y_true, y_pred, labels=[0,1])
  plt.figure()
  plt.imshow(cm, interpolation='nearest')
  plt.title(title)
  plt.xticks([0,1],["Pred 0","Pred 1"]) ; plt.yticks([0,1],["True 0","True 1"])
  for i in range(2):
      for j in range(2):
          plt.text(j, i, str(cm[i,j]), ha='center', va='center')
  plt.tight_layout()
  plt.savefig(out_path, dpi=150)
  plt.close()

def plot_feature_importance_bar(model, feature_names, title, out_path, top_n=25):
  pairs = top_feature_importance(model, feature_names, top_n=top_n)
  if not pairs:
    return
  names = [p[0] for p in pairs][::-1]  # reverse for barh
  gains = [p[1] for p in pairs][::-1]
  plt.figure(figsize=(8, max(3, len(names)*0.28)))
  plt.barh(range(len(names)), gains)
  plt.yticks(range(len(names)), names)
  plt.xlabel("Gain (importance)")
  plt.title(title)
  plt.tight_layout()
  plt.savefig(out_path, dpi=150)
  plt.close()