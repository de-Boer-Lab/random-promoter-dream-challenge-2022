import numpy as np

# FIXME scaling function not correct
def scale(value, n_original, n_buckets):
    fraction = n_buckets/n_original
    return(value * fraction + fraction/2)

def max_value(predictions,n_buckets=18, scaling=False):
    index_max = np.argmax(predictions)
    
    if scaling:
        return(scale(index_max,len(predictions),n_buckets))
    else:
        return(index_max)

def weighted_mean(predictions,n_buckets=18, scaling=False):
    value = np.average(range(0,len(predictions)),weights=predictions)

    if scaling:
        return(scale(value,len(predictions),n_buckets))
    else:
        return(value)
    