import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, precision_score, recall_score
from sklearn.metrics import mean_squared_error



def accuracy(y_true, y_pred):
    ''' Accuracy for classificaition performance '''
    acc = accuracy_score(y_true, y_pred)
    return acc

def root_mean_squared_error(y_true, y_pred):
    ''' Root Mean Squared Error for regression performance '''
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse


def classification_score(y_true, y_pred, average=None):          
    
    prec, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = None, None, None, None


    # return acc, f1_macro, f1, prec, recall, tn, fp, fn, tp
    return {
        "acc": acc, 
        "f1_macro": f1_macro, 
        "f1": f1, 
        "prec": prec, 
        "recall": recall, 
        "tn": tn, 
        "fp" : fp, 
        "fn": fn, 
        "tp": tp
        }

