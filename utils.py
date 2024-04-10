import torch.nn as nn
import torch 
import torch.nn.functional as F
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix


def AUC(y_true, y_pred):
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError as e:
        print("ValueError occurred:", e)
        auc = 0
    return auc

def Accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def Binary_classification_metrices(y_true, y_pred):
    return AUC(y_true, y_pred), Accuracy(y_true, y_pred)

"""
def Precision(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    return precision

def Recall(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    return recall

def ConfusionMatrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    return tn, fp, fn, tp

def Specificity(y_true, y_pred):
    tn, fp, fn, tp = ConfusionMatrix(y_true, y_pred)
    specificity = tn / (tn + fp)
    return specificity

def Sensitivity(y_true, y_pred):
    tn, fp, fn, tp = ConfusionMatrix(y_true, y_pred)
    sensitivity = tp / (tp + fn)
    return sensitivity



def Binary_classification_metrices(y_true, y_pred):
    return AUC(y_true, y_pred), Accuracy(y_true, y_pred), Precision(y_true, y_pred), Recall(y_true, y_pred), Specificity(y_true, y_pred), Sensitivity(y_true, y_pred)
"""


def Multiclass_classification_metrices(y_true, y_pred, num_classes):
    # Macro-averaged Precision
    macro_precision = precision_score(y_true, y_pred, average='macro')
    # Precision for each class
    precision = precision_score(y_true, y_pred, average=None)
    
    # Macro-averaged Recall
    macro_recall = recall_score(y_true, y_pred, average='macro')
    # Sensitivity/Recall for each class 
    sensitivity = recall_score(y_true, y_pred, average=None)

    # AUC :- 
    auc = []
    for i in range(num_classes):
        y_true_class = [1 if label == i else 0 for label in y_true]
        y_pred_class = [1 if pred == i else 0 for pred in y_pred]
        try:
            auc_class = roc_auc_score(y_true_class, y_pred_class)
        except ValueError as e:
            print("ValueError occurred:", e)
            auc_class = 0
        auc.append(auc_class)
    macro_AUC = sum(auc)/num_classes

    return macro_AUC, Accuracy(y_true, y_pred), macro_precision, macro_recall




def entropy(probabilities):
    # Initialize entropy value
    entropy_value = 0.0
    
    # Calculate entropy using the formula
    for prob in probabilities:
            entropy_value = entropy_value + prob * math.log2(prob + 1e-20) # Adding a small value to prevent log(0)
    
    return -entropy_value





def sample_mean_uncertainty(softmax_scores):
    # Calculate the average of softmax scores
    s = sum(softmax_scores)/len(softmax_scores)
    
    # Calculate sample mean uncertainty
    us = 1 - 2 * (s - 0.5)**2
    
    return us




class CustomMultiHeadLoss(nn.Module):
    def __init__(self, criterion, no_of_heads):
        super(CustomMultiHeadLoss, self).__init__()
        self.criterion = criterion  
        self.no_of_heads = no_of_heads

    def forward(self, head_predictions, targets):
        losses = []
        for idx in range(1, self.no_of_heads + 1):
            start_idx = 0
            end_idx = 2*idx
            pred = head_predictions[:, start_idx : end_idx]
            losses.append(self.criterion(pred, targets))
            start_idx = end_idx

        return losses
    



class CombinedMultiHeadLoss(nn.Module):
    def __init__(self, criterion, no_of_heads):
        super(CombinedMultiHeadLoss, self).__init__()
        self.criterion = criterion  
        self.no_of_heads = no_of_heads

    def forward(self, head_predictions, targets):
        losses = []
        for idx in range(1, self.no_of_heads + 1):
            start_idx = 0
            end_idx = 2*idx
            pred = head_predictions[:, start_idx : end_idx]
            losses.append(self.criterion(pred, targets))
            start_idx = end_idx

        return sum(losses)
    



class MetaMultiHeadLoss(nn.Module):
    def __init__(self, criterion, no_of_heads, epsilon):
        super(MetaMultiHeadLoss, self).__init__()
        self.criterion = criterion  
        self.no_of_heads = no_of_heads
        self.epsilon = epsilon

    def forward(self, head_predictions, targets):
        losses = []
        for idx in range(1, self.no_of_heads + 1):
            start_idx = 0
            end_idx = 2*idx
            pred = head_predictions[:, start_idx : end_idx]
            losses.append(self.criterion(pred, targets))
            start_idx = end_idx


        # Find the index with minimum value in each row (along dim=1)
        min_indices = torch.argmin(torch.stack(losses), dim=0)

        # Create a tensor with all values initialized to epsilon
        delta = torch.ones((len(head_predictions), 1),  dtype=torch.float32)*self.epsilon
        device = torch.device("cuda")
        delta = delta.to(device)

        delta[min_indices] = 1 - self.epsilon

        # Calculate the modified loss by scaling the primary criterion with delta
        modified_losses = [delta[idx] * losses[idx] for idx in range(len(losses))]

        return modified_losses
    




class MetaCombinedMultiHeadLoss(nn.Module):
    def __init__(self, criterion, no_of_heads, epsilon):
        super(MetaCombinedMultiHeadLoss, self).__init__()
        self.criterion = criterion  
        self.no_of_heads = no_of_heads
        self.epsilon = epsilon

    def forward(self, head_predictions, targets):
        losses = []
        for idx in range(1, self.no_of_heads + 1):
            start_idx = 0
            end_idx = 2*idx
            pred = head_predictions[:, start_idx : end_idx]
            losses.append(self.criterion(pred, targets))
            start_idx = end_idx


        # Find the index with minimum value in each row (along dim=1)
        min_indices = torch.argmin(torch.stack(losses), dim=0)

        # Create a tensor with all values initialized to epsilon
        delta = torch.ones((len(head_predictions), 1),  dtype=torch.float32)*self.epsilon
        device = torch.device("cuda")
        delta = delta.to(device)

        delta[min_indices] = 1 - self.epsilon

        # Calculate the modified loss by scaling the primary criterion with delta
        modified_losses = [delta[idx] * losses[idx] for idx in range(len(losses))]

        return sum(modified_losses)


def stochastic_dropout(predictions, threshold=0.1):
    # Generate random mask based on the probability threshold
    mask = torch.rand_like(predictions) < threshold
    
    # Apply the mask to the predictions
    masked_predictions = predictions * mask
    
    return masked_predictions


def min_max_normalization(List):
    # Calculate the min and max values of the list
    min_val = min(List)
    max_val = max(List)
    # Normalize each entropy value to the range [0, 1]
    normalized_list = [(x - min_val) / (max_val - min_val) for x in List]

    return normalized_list
    



    



    

        

