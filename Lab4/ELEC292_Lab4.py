import matplotlib as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Question 1
def Q1():
    dataset = pd.read_csv("heart.csv")
    data = dataset.iloc[:, :-1]  
    labels = dataset.iloc[:, -1]
    fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20, 10))
    data.hist(ax=ax.flatten()[:13])

# Question 3
def Q3():
    dataset = pd.read_csv("heart.csv")
    data = dataset.iloc[:, :-1]  
    labels = dataset.iloc[:, -1]
    fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20, 10))
    for i in range(0, 13):
        ax.flatten()[i].hist(data.iloc[:, i])
        ax.flatten()[i].set_title(data.columns[i], fontsize=15)

# Question 4
def Q4():
    dataset = pd.read_csv("heart.csv")
    data = dataset.iloc[:, :-1]  
    labels = dataset.iloc[:, -1]
    fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20, 10))
    data.plot(ax=ax.flatten()[:13], kind='density', subplots=True, sharex=False)

# Question 5
def Q5():
    dataset = pd.read_csv("heart.csv")
    data = dataset.iloc[:, :-1]  
    labels = dataset.iloc[:, -1]
    fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20, 10))
    data.plot(ax=ax.flatten()[:13], kind='box', subplots=True, sharex=False, sharey=False)

# Question 6
def Q6():
    dataset = pd.read_csv("heart.csv")
    data = dataset.iloc[:, :-1]  
    labels = dataset.iloc[:, -1]
    fig, ax = plt.subplots(ncols=13, nrows=13, figsize=(30, 30))
    pd.plotting.scatter_matrix(data, ax=ax)

# Question 8

def Q89():
    dataset = pd.read_csv("wine.csv")
    dataset = dataset.drop(dataset.columns[0], axis=1)

    dataset['quality'] = dataset['quality'].apply(lambda x: 1 if x>= 8 else 0)

    dataset = dataset.sort_values(by='quality')
    data = dataset.iloc[:, :-1]
    labels = dataset.iloc[:, -1]
    sc = StandardScaler()
    pca = PCA(n_components=2)
    data = sc.fit_transform(data)
    data = pca.fit_transform(data)

    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], c=dataset['quality'].apply(lambda x: 'red' if x==1 else 'pink'))
    plt.title('PCA 1v2')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    data = dataset.iloc[:, :-1]
    labels = dataset.iloc[:, -1]
    data = sc.fit_transform(data)
    tsne = TSNE(n_components=2, perplexity=30)
    data = tsne.fit_transform(data)

    plt.figure(figsize=(10,10))
    plt.scatter(data[:, 0], data[:, 1], c=dataset['quality'].apply(lambda x: 'red' if x==1 else 'pink'))
    plt.title('t-SNE 1v2')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    data = dataset.iloc[:, :-1]
    labels = dataset.iloc[:, -1]
    sc = StandardScaler()
    pca = PCA(n_components=11)
    data = sc.fit_transform(data)
    data = pca.fit_transform(data)

    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 7], data[:, 8], c=dataset['quality'].apply(lambda x: 'red' if x==1 else 'pink'))
    plt.title('PCA 8v9')
    plt.xlabel('Principal Component 8')
    plt.ylabel('Principal Component 9')

# Function Calls
Q1()
Q3()
Q4()
Q5()
Q6()
Q89()

plt.show()

