import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA

def split():
    dataset = pd.read_csv("winequalityN-lab6.csv") # read dataset
    dataset = dataset.drop(dataset.columns[0], axis=1) # drop wine type
    dataset['quality'] = dataset['quality'].apply(lambda x: 1 if x >= 6 else 0) # set quality to binary

    data = dataset.iloc[:, :-1] # data
    labels = dataset.iloc[:, -1] # labels

    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=0) # split with 0 randomness
    return X_train, X_test, Y_train, Y_test

def Part1(X_train, X_test, Y_train, Y_test):
    scaler = StandardScaler() # initialize scaler
    l_reg = LogisticRegression(max_iter=10000) # initialize logistic regression

    clf = make_pipeline(scaler, l_reg) # make pipeline clf
    clf.fit(X_train, Y_train) # fit pipeline

    y_pred = clf.predict(X_test) # prediction
    y_clf_prob = clf.predict_proba(X_test) # probability
    print('y_pred is:', y_pred)
    print('y_clf_prob is:', y_clf_prob)

    acc = accuracy_score(Y_test, y_pred) # accuracy
    print('accuracy is:', acc)

    recall = recall_score(Y_test, y_pred) # recall
    print('recall is:', recall)

    cm = confusion_matrix(Y_test, y_pred) # confusion matrix
    cm_display = ConfusionMatrixDisplay(cm).plot() # plot cm
    plt.show()

    f1 = f1_score(Y_test, y_pred) # f1 score
    print('the f1 score is:', f1)

    auc = roc_auc_score(Y_test, y_clf_prob[:, 1]) # AUC
    print('the AUC is:', auc)

    fpr, tpr, thresholds = roc_curve(Y_test, y_clf_prob[:, 1], pos_label=clf.classes_[1]) # FPR/TPR
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc).plot() # plot ROC
    plt.show()

def Part2(X_train, X_test, Y_train, Y_test):
    scaler = StandardScaler() # initialize scaler
    l_reg = LogisticRegression(max_iter=10000) # initialize logistic regression
    pca = PCA(n_components=2) # initialize PCA

    pca_pipe = make_pipeline(scaler, pca) # make pipeline for PCA
    pca_pipe.fit(X_train, Y_train) # fit pipeline

    X_train_pca = pca_pipe.transform(X_train) # transform X train
    X_test_pca = pca_pipe.transform(X_test) # transform X test

    clf = make_pipeline(l_reg) # make clf pipeline
    clf.fit(X_train_pca, Y_train) # fit pipeline

    y_pred_pca = clf.predict(X_test_pca) # predict for PCA

    disp = DecisionBoundaryDisplay.from_estimator(
        clf, X_train_pca, response_method="predict",
        xlabel='X1', ylabel='X2',
        alpha=0.5,
    ) # display boundary object from an estimator using the clf pipeline as the classifier, X_train_pca (used to train clf) as what decision boundaries will be calculated off, and the response method meaning the classes are to be predicted using the classifier (clf pipeline)
    disp.ax_.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=Y_train) # scatter plot, where the first principal component is the first column of the training data, and the second principal component is the second column of the training data
    plt.show()

    acc = accuracy_score(Y_test, y_pred_pca) # accuracy
    print('accuracy is:', acc)

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = split()
    Part1(X_train, X_test, Y_train, Y_test)
    print('\n')
    Part2(X_train, X_test, Y_train, Y_test)
    # in summary, Part 2 is substantially more inaccurate than Part 1, by a difference of anywhere between 5-15%