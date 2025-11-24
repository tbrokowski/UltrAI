
from sklearn.metrics import roc_auc_score, accuracy_score, \
    recall_score, confusion_matrix, \
    balanced_accuracy_score, roc_curve, f1_score
import numpy as np
import warnings
from sklearn.metrics import precision_recall_curve, auc
# def compute_metrics(targets, logits):
#     preds = [float(i > 0) for i in logits]    #original way
#     #preds = [np.round(1/(1+np.exp(-i))) for i in logits]   #my way (identical)
#     tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
#     fpr, tpr, _ = roc_curve(y_true=targets, y_score=logits)

#     metric_dict = {'accuracy': accuracy_score(targets, preds),
#                    'sensitivity': recall_score(targets, preds, pos_label=1),
#                    'specificity': recall_score(targets, preds, pos_label=0),
#                    'balanced_accuracy': balanced_accuracy_score(targets, preds),
#                    'roc_auc': roc_auc_score(targets, logits),
#                    'f1_score': f1_score(targets, preds),
#                    'true_positive': tp,
#                    'false_positive': fp,
#                    'true_negative': tn,
#                    'false_negative': fn,
#                    'false_positive_rate': fpr,
#                    'true_positive_rate': tpr}
#     return metric_dict


def compute_metrics_final(targets, logits):
    preds = [float(i > 0) for i in logits]    #original way
    #preds = [np.round(1/(1+np.exp(-i))) for i in logits]   #my way (identical)
    probabilities = 1 / (1 + np.exp(-logits))
    
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
    fpr, tpr, _ = roc_curve(y_true=targets, y_score=logits)

    precision, recall, _ = precision_recall_curve(targets, probabilities)
    auprc = auc(recall, precision)

    preds_0 = [float(i > 0.5) for i in logits] 
    preds_1 = [float(i > 1.0) for i in logits] 
    preds_2 = [float(i > 2.0) for i in logits] 

    metric_dict = {'accuracy': accuracy_score(targets, preds),
                   'sensitivity': recall_score(targets, preds, pos_label=1),
                   'specificity': recall_score(targets, preds, pos_label=0),
                   'balanced_accuracy': balanced_accuracy_score(targets, preds),
                   'roc_auc': roc_auc_score(targets, logits),
                   'auprc': auprc,  
                   'f1_score': f1_score(targets, preds),
                   'f1_score_1': f1_score(targets, preds_0),
                    'f1_score_2': f1_score(targets, preds_1),
                    'f1_score_3': f1_score(targets, preds_2),
                   'true_positive': tp,
                   'false_positive': fp,
                   'true_negative': tn,
                   'false_negative': fn,
                 #  'false_positive_rate': fpr,
                 #  'true_positive_rate': tpr
                 }

    return metric_dict


def compute_metrics(targets, logits):
    # Check if targets are one-hot encoded (multiclass) or not (binary)
    #print(targets, logits)
    targets = np.array(targets)
    logits = np.array(logits)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    #print("Targets Shape", targets.shape, "Logits shape", logits.shape)
    default_metric_dict = {
        'accuracy': 0.0,
        'sensitivity': 0.0,
        'specificity': 0.0,
        'balanced_accuracy': 0.0,
        'roc_auc': 0.0,
        'auprc': 0.0, 
        'f1_score': 0.0,
        'precision': 0.0,
        'confusion_matrix': None
    }

    if targets.ndim == 1 or targets.shape[1] == 1:
        # Binary classification
        #preds = (logits > 0).astype(float)
        probabilities = 1 / (1 + np.exp(-logits))
        preds = (probabilities > 0.5).astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
            fpr, tpr, _ = roc_curve(targets, probabilities)
            precision, recall, _ = precision_recall_curve(targets, probabilities)
            auprc = auc(recall, precision)

            metric_dict = {
                'accuracy': accuracy_score(targets, preds),
                'sensitivity': recall_score(targets, preds, pos_label=1),
                'specificity': recall_score(targets, 1 - preds, pos_label=1),  # Calculating specificity
                'balanced_accuracy': balanced_accuracy_score(targets, preds),
                'roc_auc': roc_auc_score(targets, probabilities),
                'auprc': auprc,  # Added AUPRC
                'f1_score': f1_score(targets, preds),
                'confusion_matrix': (tn, fp, fn, tp),
              #  'false_positive_rate': fpr,
               # 'true_positive_rate': tpr
            }
        except ValueError as e:
            print(f"Error in metric computation: {e}")
            return {
                'accuracy': 0.0,
                'sensitivity': 0.0,
                'specificity': 0.0,
                'balanced_accuracy': 0.0,
                'roc_auc': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'confusion_matrix': None,
              #  'false_positive_rate': None,
               # 'true_positive_rate': None
            }

    else:
        # Multiclass classification
        num_classes = logits.shape[1]
        preds = np.argmax(logits, axis=1)
        targets = np.argmax(targets, axis=1) if targets.ndim > 1 else targets

        try:
            # Compute per-class metrics
            confusion = confusion_matrix(targets, preds, labels=range(num_classes))
            sensitivity = recall_score(targets, preds, average='macro')
            specificity = recall_score(targets, preds, average='macro', pos_label=0)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fpr, tpr, _ = roc_curve(targets.ravel(), probabilities.ravel(), pos_label=1) if num_classes == 2 else (None, None, None)
            
            metric_dict = {
                'accuracy': accuracy_score(targets, preds),
                'sensitivity': float(sensitivity),  # Convert to native type
                'specificity': float(specificity),  # Convert to native type
                'balanced_accuracy': balanced_accuracy_score(targets, preds),
                'roc_auc': roc_auc_score(targets, probabilities, multi_class='ovr'),
                'f1_score': f1_score(targets, preds, average='weighted'),
                'confusion_matrix': confusion.tolist()  # Add confusion matrix to metrics
            }
        except ValueError as e:
            print(f"Skipping batch due to error: {e}")
            return default_metric_dict

    return metric_dict

class BestMetricVal:
    """Keep only the best value (min or max)."""

    def __init__(self, objective: str):
        if objective not in ["min", "max"]:
            raise ValueError(f"Objective should be 'min' or 'max', not '{objective}'.")
        self.value = None
        self.objective = objective

    def append(self, new_value):
        is_new_best = False
        if self.value is None:
            is_new_best = True
        else:
            if self.objective == "min" and new_value < self.value:
                is_new_best = True
            elif self.objective == "max" and new_value > self.value:
                is_new_best = True

        if is_new_best:
            self.value = new_value

        return is_new_best

