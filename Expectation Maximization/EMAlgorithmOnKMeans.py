
## Expectation Maximum Algorithm
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# load dataset
dataframe = pd.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

#Define number of clusters
n_clusters = 3

# 1. Randomly choose clusters
rng = np.random.RandomState(42)
i = rng.permutation(X.shape[0])[:n_clusters]
centers = X[i]
plt.title('Randomly Choosing Gaussians(Clusters)')
plt.scatter(X[:, 0], X[:, 1], c='gray', s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=200, alpha=0.5);

count = 1

 while True:
        # E-step
        # 2a. Assign tags based on closest center
        tags = pairwise_distances_argmin(X, centers)

        # M-step
        # 2b. Find new centers from means of points
        new_centers = np.array([X[tags == i].mean(0)
                                for i in range(n_clusters)])
        plt.figure()
        plt.title('Iteration {}: E-step and M-step'.format(count))
        plt.scatter(X[:, 0], X[:, 1], c=tags, s=50, cmap='viridis')
        plt.scatter(new_centers[:, 0], new_centers[:, 1], c='blue', s=200, alpha=0.5);
        plt.figtext(0.03, -0.03,'Centers: {} , {}'.format(new_centers[:, 0], new_centers[:, 1]) )
        count = count + 1
        # 2c. Check for convergence
        if np.all(centers == new_centers):

            break
        #Finish iteration
        centers = new_centers


# Iterations end with the following clusters
plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=200, alpha=0.5);
plt.scatter(X[:, 0], X[:, 1], c=tags,s=50, cmap='viridis');
plt.title('Final Clusters')
