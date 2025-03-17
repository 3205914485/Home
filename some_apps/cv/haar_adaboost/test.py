import numpy as np

def find_best_threshold(feature_values, labels, weights):
    # sort by f_v
    sorted_idx = np.argsort(feature_values)
    sorted_feature_values = feature_values[sorted_idx]
    sorted_labels = labels[sorted_idx]
    sorted_weights = weights[sorted_idx]

    T1 = np.sum(sorted_weights[sorted_labels == 1]) # pos sum
    T0 = np.sum(sorted_weights[sorted_labels == -1]) # neg sum


    S1, S0 = 0, 0

    min_error = np.inf
    best_threshold = None
    best_polarity = None

    for i in range(1, len(sorted_feature_values)):
        # update S0 & S1:
        if sorted_labels[i-1] == 1:
            S1 += sorted_weights[i-1]
        else:
            S0 += sorted_weights[i-1]
        
        error1 = S1 + (T0 - S0)  # < threshold --> neg 
        error2 = S0 + (T1 - S1)  # > threshold --> pos
        error = min(error1, error2)
        
        if error < min_error:
            min_error = error
            best_threshold = (sorted_feature_values[i-1] + sorted_feature_values[i]) / 2
            best_polarity = -1 if error2 < error1 else 1

    return best_threshold, best_polarity, min_error


def search_learner(features, labels, weights):
    best_overall_threshold = None
    best_overall_polarity = None
    lowest_error = np.inf
    best_feature_index = None

    for feature_index, feature_values in enumerate(features):
        threshold, polarity, error = find_best_threshold(feature_values, labels, weights)
        if error < lowest_error:
            best_overall_threshold = threshold
            best_overall_polarity = polarity
            lowest_error = error
            best_feature_index = feature_index


    return best_feature_index, best_overall_threshold, best_overall_polarity, lowest_error

features = np.array([[1,2,3,4,5]])
labels = np.array([-1,-1,1,1,-1])
weights = np.array([1,1,1,1,1])
best_feature, best_threshold, best_polarity, error = search_learner(features, labels, weights)