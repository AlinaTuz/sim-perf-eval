# General structure of a data point
# data_point = [(x1, x2, ..., xn), predicted_value, actual_value]

# Confusion matrix

def confusion(data):
    """
    Takes as a parameter a list containing data points formatted the following way:
    data_point = [(x1, x2, ..., xn), predicted_value, actual_value]

    --> We admit here that both predicted and actual values are boolean.

    Returns the number of true positives, false negatives, false positives, and true negatives
    """

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for _, predicted_value, actual_value in data:
        if predicted_value:
            if actual_value:
                tp += 1
            else:
                fp += 1
        else:
            if actual_value:
                fn += 1
            else:
                tn += 1

    return tp, fn, fp, tn


# Accuracy

def accuracy(data):
    tp, _, _, tn = confusion(data)
    return (tp + tn) / len(data)

# Precision

def precision(data):
    tp, _, fp, _ = confusion(data)
    return tp / (tp + fp)

# Recall

def recall(data):
    tp, fn, _, _ = confusion(data)
    return tp / (tp + fn)

# F1 score

def f1(data):
    p = precision(data)
    r = recall(data)
    return 2*p*r / (p + r)