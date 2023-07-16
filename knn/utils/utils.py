import numpy as np

def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray):
    return np.linalg.norm(np.array(vec1)-np.array(vec2))

def standardize(data: np.mat) -> np.mat:
    data_normed = (data - data.mean(axis=0))/(data.std(axis=0))
    return data_normed
