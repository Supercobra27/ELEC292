# Author(s): Ryan Silverberg
import pandas as pd
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,roc_curve,roc_auc_score,RocCurveDisplay
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

# Global Variables
scaler = StandardScaler()

csvs = [
    "RyanWalk.csv",
    "RyanJump.csv",
    "FosterWalk.csv",
    "FosterJump.csv",
    "JuliaWalk.csv",
    "JuliaJump.csv"
]

data_list = []

# Remove pre-existing HDF5 file to prevent overlap
if os.path.exists("main_data.hdf5"):
    os.remove("main_data.hdf5")

# Remove pre-existing features file to prevent overlap
if os.path.exists("features.csv"):
    os.remove("features.csv")

def segment(dataset, window_size=5):
    # Calculate the number of windows
    num_windows = int(dataset['Time (s)'].iloc[-1] // window_size) + 1

    # Initialize an empty list to hold the dataframes
    grouped = []

    # Loop over each window
    for i in range(num_windows):
        # Get the start and end times for this window
        start_time = i * window_size
        end_time = (i + 1) * window_size

        # Create a new dataframe for this window
        window_df = dataset[(dataset['Time (s)'] >= start_time) & (dataset['Time (s)'] < end_time)].copy()
        # Append the new dataframe to the list
        grouped.append(window_df)

    return grouped

def shuffle(data):
    np.random.shuffle(data)
    return data

def label(csv, label):
    data = pd.read_csv(csv)
    data["label"] = label
    return data

for csv in csvs:
    if csv.find("Walk") != -1:
        data_list.append(label(csv, 0))
    elif csv.find("Jump") != -1:
        data_list.append(label(csv, 1))

m1walk = data_list[0]
m1jump = data_list[1]
m2walk = data_list[2]
m2jump = data_list[3]
m3walk = data_list[4]
m3jump = data_list[5]

total_data = [
    m1walk,
    m1jump,
    m2walk,
    m2jump,
    m3walk,
    m3jump
]

total_walk = [
    m1walk, 
    m2walk, 
    m3walk
]

total_jump = [
    m1jump, 
    m2jump, 
    m3jump
]

# Combine all relevant datasets into dataframes
combined_df = pd.concat(total_data, ignore_index=True)
walk_df = pd.concat(total_walk, ignore_index=True)
jump_df = pd.concat(total_jump, ignore_index=True)

# Check NaN Amounts
print("Checking NaN values within dataframe...\n")
combined_df = combined_df.iloc[:, 0:6]
print(combined_df.isna().sum())
print("\nNaN Values within dataframe after dropna()...\n")
combined_df = combined_df.dropna()
print(combined_df.isna().sum())

segmented_data_walking = segment(walk_df)
segmented_data_jumping = segment(jump_df)
segmented_data = segmented_data_walking
segmented_data.extend(segmented_data_jumping)

final_data = pd.concat(shuffle(segmented_data), ignore_index=True).iloc[:, 0:6]

# Split the data into Training (90%) and Testing (10%)
X_train, X_test, Y_train, Y_test = train_test_split(final_data, final_data.iloc[:, 5], test_size=0.1, random_state=0)

with h5py.File("main_data.hdf5", 'a') as hdf:
    i = 0

    # Create Member Groups
    g1 = hdf.create_group('/member1')
    g2 = hdf.create_group('/member2')
    g3 = hdf.create_group('/member3')

    # Create Training Group
    train_group = hdf.create_group('/model')
    train_group.create_dataset('train', data=X_train)
    train_group.create_dataset('test', data=Y_train)

    for csv in csvs:
        if i < 2:
            group = g1
        elif i >= 2 and i < 4:
            group = g2
        elif i >= 4:
            group = g3

        if csv.find("Walk") != -1:
            data = label(csv, 0)
            name = "walk"
        elif csv.find("Jump") != -1:
            data = label(csv, 1)
            name = "jump"
        group.create_dataset(name, data=data)
        i += 1


# Visualization
def plot_motion(data, activity, name, window_size=5):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(data.iloc[:, 0], data.iloc[:, -5].rolling(window_size).mean(),
                 label="Linear Acceleration x (m/s^2)")
    axes[0].set_ylabel("X Acceleration (m/s^2)")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(data.iloc[:, 0], data.iloc[:, -4].rolling(window=window_size).mean(),
                 label="Linear Acceleration y (m/s^2)", color='orange')
    axes[1].set_ylabel("Y Acceleration (m/s^2)")
    axes[1].legend()
    axes[1].grid(True)
    axes[2].plot(data.iloc[:, 0], data.iloc[:, -3].rolling(window=window_size).mean(),
                 label="Linear Acceleration z (m/s^2)", color='green')
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Z Acceleration (m/s^2)")
    axes[2].legend()
    axes[2].grid(True)
    axes[3].plot(data.iloc[:, 0], data.iloc[:, -2].rolling(window=window_size).mean(),
                 label="Total Acceleration (m/s^2)", color='red')
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Total Acceleration (m/s^2)")
    axes[3].legend()
    axes[3].grid(True)
    plt.suptitle("Acceleration VS Time for "+name+" | "+activity, fontsize=20)

def plot_average(data, activity, name):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    return 1

def plot_histogram(data, activity, name):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    return 1

def plot_segmented(data, slice=1):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    return 1

i = 0
for data in total_data:
    if "Walk" in csvs[i]:
        activity = 'walk'
    elif "Jump" in csvs[i]:
        activity = 'jump'
    if i < 2:
        name = "member1"
    elif i >= 2 and i < 4:
        name = "member2"
    elif i >= 4:
        name = "member3"
    #plot_motion(data, activity, name)
    #plot_average(data, activity, name)
    #plot_histogram(data, activity, name)
    i += 1

# Plot Segmented Slice here
# plot_segmented(final_data)
plt.show()

# Preprocessing
final_data = final_data.iloc[:, 0:4]
final_data = final_data.rolling(window=5).mean()

# Feature Extraction
window_size = 5
features = pd.DataFrame()
final_features = final_data
for column in final_features.columns:
    feature_column = pd.DataFrame()
    feature_column['mean'] = final_features[column].rolling(window=window_size).mean()
    feature_column['std'] = final_features[column].rolling(window=window_size).std()
    feature_column['max'] = final_features[column].rolling(window=window_size).max()
    feature_column['min'] = final_features[column].rolling(window=window_size).min()
    feature_column['kurtosis'] = final_features[column].rolling(window=window_size).kurt()
    feature_column['skew'] = final_features[column].rolling(window=window_size).skew()
    feature_column['median'] = final_features[column].rolling(window=window_size).median()
    feature_column['z-score'] = (final_features[column]-feature_column['mean']) / feature_column['std']
    feature_column['range'] = feature_column['max'] - feature_column['min']
    feature_column['variance'] = final_features[column].rolling(window=window_size).var()
    features = pd.concat([features, feature_column], axis=1).dropna()
    # final_features[column] = final_features[column] - feature_column['mean']
    # final_features[column] = final_features[column] / feature_column['std']

# Export features to a csv (Cause it's cool)
features.to_csv('features.csv', index=False)

print(features)

# Normalization
final_data = final_data.dropna()
final_data = scaler.fit_transform(final_data)
print(final_data)

# Classification
model = LogisticRegression(max_iter=10000)
clf = make_pipeline(scaler, model) # make pipeline clf
clf.fit(X_train.dropna(), Y_train.dropna()) # fit pipeline

Y_pred = clf.predict(X_test.dropna()) # prediction
y_clf_prob = clf.predict_proba(X_test.dropna()) # probability
print('Y_pred is:', Y_pred)
print('y_clf_prob is:', y_clf_prob)

acc = accuracy_score(Y_test, Y_pred) # accuracy
print('accuracy is:', acc)

recall = recall_score(Y_test, Y_pred) # recall
print('recall is:', recall)

cm = confusion_matrix(Y_test, Y_pred) # confusion matrix
cm_display = ConfusionMatrixDisplay(cm).plot() # plot cm
plt.show()
auc = roc_auc_score(Y_test, y_clf_prob[:, 1]) # AUC
print('the AUC is:', auc)

fpr, tpr, thresholds = roc_curve(Y_test, y_clf_prob[:, 1], pos_label=clf.classes_[1]) # FPR/TPR
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc).plot() # plot ROC
plt.show()