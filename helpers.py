from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import math
import matplotlib.pyplot as plt
import numpy as np


def get_matrix_from_features(X_new, X_old):
    X_old_norm = normalize(X_old, norm='l1', axis=1,
                           copy=True, return_norm=False)
    X_new_norm = normalize(X_new, norm='l1', axis=1,
                           copy=True, return_norm=False)

    return cosine_similarity(X_old_norm, X_new_norm)


def get_matrix_from_model(X_new, X_old, estimator):
    matrix = np.zeros((X_new.shape[0], X_old.shape[0]))

    for i in range(X_new.shape[0]):
        for j in range(X_old.shape[0]):
            matrix[i][j] = estimator.predict_proba(np.concatenate(
                [X_new[i], X_old[j]]).reshape(1, -1))[0][1].reshape(1, -1)

    return matrix


def d_prime_with_plt(S, filename):
    diagonal = S.diagonal()
    non_diagonal = S[np.where(~np.eye(S.shape[0], dtype=bool))]

    mu_1, sigma_1 = np.mean(diagonal), np.std(diagonal)
    mu_2, sigma_2 = np.mean(non_diagonal), np.std(non_diagonal)

    d_prime = math.fabs(mu_2 - mu_1) / math.sqrt((sigma_1**2 + sigma_2**2) / 2)

    plt.figure(figsize=(10, 5))

    plt.hist(diagonal, 100, label='genuine', histtype='step',
             density=True, color='green', stacked=True)
    plt.hist(non_diagonal, 100, label='impostor', histtype='step',
             density=True, color='red', stacked=True)

    plt.legend()
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')

    plt.savefig(filename, dpi=300, pad_inches=0.1, bbox_inches='tight')
    plt.show()

    return d_prime
