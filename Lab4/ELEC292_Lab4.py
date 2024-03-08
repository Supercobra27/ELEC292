import matplotlib as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Question 1
dataset = pd.read_csv("heart.csv")
data = dataset.iloc[:, :-1]  
labels = dataset.iloc[:, -1]
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20, 10))

data.hist(ax=ax.flatten()[:13])

# Question 2



# Question 3
dataset = pd.read_csv("heart.csv")
data = dataset.iloc[:, :-1]  
labels = dataset.iloc[:, -1]
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20, 10))
for i in range(0, 13):
    ax.flatten()[i].hist(data.iloc[:, i])
    ax.flatten()[i].set_title(data.columns[i], fontsize=15)

# Question 4
dataset = pd.read_csv("heart.csv")
data = dataset.iloc[:, :-1]  
labels = dataset.iloc[:, -1]
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20, 10))
data.plot(ax=ax.flatten()[:13], kind='density', subplots=True, sharex=False)

# Question 5
dataset = pd.read_csv("heart.csv")
data = dataset.iloc[:, :-1]  
labels = dataset.iloc[:, -1]
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20, 10))
data.plot(ax=ax.flatten()[:13], kind='box', subplots=True, sharex=False, sharey=False)

# Question 6
dataset = pd.read_csv("heart.csv")
data = dataset.iloc[:, :-1]  
labels = dataset.iloc[:, -1]
fig, ax = plt.subplots(ncols=13, nrows=13, figsize=(30, 30))
pd.plotting.scatter_matrix(data, ax=ax)

# Question 7

# Question 8
dataset = pd.read_csv("wine.csv")
dataset = dataset.drop(dataset.columns[0], axis=1)

def map_quality(quality):
    if quality >= 8:
        return 1
    else:
        return 0

# Apply the function to the 'quality' column
dataset['quality'] = dataset['quality'].apply(map_quality)
def map_quality_to_color(quality):
    if quality == 1:
        return 'red'  # High-quality wines
    else:
        return 'pink'  # Low-quality wines

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(dataset.drop('quality', axis=1))

# Plot PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=dataset['quality'].apply(map_quality_to_color))
plt.title('PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30)
tsne_result = tsne.fit_transform(dataset.drop('quality', axis=1))

# Plot t-SNE results
plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=dataset['quality'].apply(map_quality_to_color))
plt.title('t-SNE')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
# Output
fig.tight_layout()
plt.show()


