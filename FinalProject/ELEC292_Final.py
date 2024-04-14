# Author(s): Ryan Silverberg, Foster Ecklund, Julia Zigelstein
from tkinter import messagebox
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
import tkinter as tk
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from tkinter import filedialog

# Global Variables
scaler = StandardScaler()
#Global Variables for App
canvas = None
csv_out = None

csvs = [
    "RyanWalk.csv",
    "RyanJump.csv",
    "FosterWalk.csv",
    "FosterJump.csv",
    "JuliaWalk.csv",
    "JuliaJump.csv"
]

filepaths = [
    "main_data.hdf5",
    "features.csv"
]

data_list = []

for file in filepaths:
    if os.path.exists(file):
        os.remove(file)

def segment(dataset, window_size=5):
    # Calculate window numbers
    num_windows = int(dataset['Time (s)'].iloc[-1] // window_size) + 1
    segments = []

    # go over each iter
    for i in range(num_windows):
        # Get time lengths
        start = i * window_size
        end = (i + 1) * window_size
        window_df = dataset[(dataset['Time (s)'] >= start) & (dataset['Time (s)'] < end)].copy()
        # Add it to the list
        segments.append(window_df)

    return segments

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

categories = [
    "Time (s)",
    "Linear Acceleration x (m/s^2)",
    "Linear Acceleration y (m/s^2)",
    "Linear Acceleration z (m/s^2)",
    "Absolute acceleration (m/s^2)", 
    "label"
]

# Combine all relevant datasets into dataframes
combined_df = pd.concat(total_data, ignore_index=True)
walk_df = pd.concat(total_walk, ignore_index=True)
jump_df = pd.concat(total_jump, ignore_index=True)

# Check NaN Amounts
print("Checking NaN values within dataframe...\n")
combined_df = combined_df.iloc[:, 0:6]
print(combined_df.iloc[:, 0:5].isna().sum())
print("\nNaN Values within dataframe after dropna()...\n")
combined_df = combined_df.dropna()
print(combined_df.iloc[:, 0:5].isna().sum())

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
                 label="Linear Acceleration X (m/s^2)")
    axes[0].set_ylabel("X Acceleration (m/s^2)")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(data.iloc[:, 0], data.iloc[:, -4].rolling(window=window_size).mean(),
                 label="Linear Acceleration Y (m/s^2)", color='orange')
    axes[1].set_ylabel("Y Acceleration (m/s^2)")
    axes[1].legend()
    axes[1].grid(True)
    axes[2].plot(data.iloc[:, 0], data.iloc[:, -3].rolling(window=window_size).mean(),
                 label="Linear Acceleration z (m/s^2)", color='green')
    axes[2].set_ylabel("Z Acceleration (m/s^2)")
    axes[2].legend()
    axes[2].grid(True)
    axes[3].plot(data.iloc[:, 0], data.iloc[:, -2].rolling(window=window_size).mean(),
                 label="Total Acceleration (m/s^2)", color='red')
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Total Acceleration (m/s^2)")
    axes[3].legend()
    axes[3].grid(True)
    plt.suptitle("Acceleration VS Time for "+name+" - "+activity, fontsize=20)
    return fig

def plot_average(data, activity, name):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    return 1

def plot_histogram(data, activity, name):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    return 1

def plot_segmented(data, slice=1):
    data_slice = data.sample(n=slice)

    # Normalize Time Slice to not hinder data visualization
    data_slice["Time (s)"] = data_slice["Time (s)"]/10
    plt.figure(figsize=(20, 10))
    plt.bar(categories[:5], data_slice.iloc[0, 0:5].tolist())
    plot_motion(data_slice, 'none', 'shuffled')

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

#plot_segmented(final_data)
plt.show()

# Preprocessing
final_data = final_data.iloc[:, 0:4]
final_data = final_data.rolling(window=5).mean()

# Feature Extraction
def feature_extraction(final_data, window_size=5):
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
        # Add more features
        return features

    # Actual z-score calculation/standardization
    # final_features[column] = final_features[column] - feature_column['mean']
    # final_features[column] = final_features[column] / feature_column['std']

# Export features to a csv (Cause it's cool)
# features.iloc[:, :10].to_csv('features.csv', index=False)
    
features = feature_extraction(final_data)
print(f"Features Shape --> {features.shape}")
print(f"\nFeatures (mean): \n{features.mean()}\n")

# Normalization
X_train = feature_extraction(X_train)
X_test = feature_extraction(X_test)

final_data = final_data.dropna()
fit = scaler.fit(X_train)
X_train = fit.transform(X_train)
X_test = fit.transform(X_test)
Y_train = Y_train.values.flatten()
Y_test = Y_test.values.flatten()
print(f"Mean after normalzation is {round(X_train.mean())}\n")

# Classification
model = LogisticRegression(max_iter=10000)
clf = make_pipeline(scaler, model)

# Trimming
len_X = len(X_train)
len_Y = len(Y_train)
min_len = min(len_X, len_Y)
X_train = X_train[:min_len]
Y_train = Y_train[:min_len]

clf.fit(X_train, Y_train)

# Predictions
Y_pred = clf.predict(X_test)
Y_clf_prob = clf.predict_proba(X_test)

# Print numpy array shapes
print('Y_pred:', Y_pred)
print('Y_clf_prob:', Y_clf_prob)
print('\nY_pred Shape -->', Y_pred.shape)
print('Y_clf_prob Shape -->', Y_clf_prob.shape)

# Transfer Data to HDF5
with h5py.File("main_data.hdf5", 'a') as hdf:
    analyzed = hdf.create_group('/analysis')
    pred_df = pd.DataFrame(Y_pred)
    clf_pdf = pd.DataFrame(Y_clf_prob)
    Y_out = pd.concat([pred_df, clf_pdf], axis=1)
    analyzed.create_dataset('predictions', data=Y_out)
    analyzed.create_dataset("features",data=features)

    

# Calculate prediction output
len_X = len(Y_test)
len_Y = len(Y_pred)
min_len = min(len_X, len_Y)
Y_test = Y_test[:min_len]
Y_pred = Y_pred[:min_len]
accuracy = accuracy_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
area_under_curve = roc_auc_score(Y_test, Y_clf_prob[:, 1])

# Print prediction outputs
print('\nAccuracy:', accuracy)
print('Recall:', recall)
print('AUC:', area_under_curve)

# Print Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

# Print Accuracy Curve
fpr, tpr, thresholds = roc_curve(Y_test, Y_clf_prob[:, 1], pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=area_under_curve).plot()
plt.show()

#create and return a graph from a CSV file
def create_graph(csv_file):
    #need to add graph for 1/0 walking/jumping
    df = pd.read_csv(csv_file)
    df2 = df
    df2_time = df2.iloc[:, 0]
    fig = plot_motion(df2, 'example', os.path.basename(csv_file))
    # ax.plot(df["Time (s)"], df["Linear Acceleration z (m/s^2)"])
    # ax.set_title("Graph from " + os.path.basename(csv_file))
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Linear Acceleration z (m/s^2)")
    # ax.set_ylim([-75, 75])
    return fig, df

# Function to update the displayed graph
def load_file():
    global canvas, csv_in, csv_out
    csv_in = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    csv_out = csv_in

    if canvas:
        canvas.get_tk_widget().destroy()  # Destroy the previous canvas
    fig, csv_in = create_graph(csv_out)
    #call classifier
    csv_in = pd.read_csv(csv_out)

   # csv_in.drop(["Linear Acceleration x (m/s^2)",
    #"Linear Acceleration y (m/s^2)","Absolute acceleration (m/s^2)","label"])
    csv_segments = segment(csv_in)
    for window in csv_segments:
        window.drop(["Time (s)"], axis=1, inplace=True)
    
    sma = [window.rolling(5).mean().dropna() for window in csv_segments]
    sma_features = feature_extraction(sma)
    sma_norm = fit.transform(sma_features)

    prediction = model.predict(sma_norm) # Put DataFrame in here
    print(f"\nPrediction: {prediction}")
    canvas = FigureCanvasTkAgg(fig, master=app)
    canvas.draw()
    canvas.get_tk_widget().pack()

def save_file():
    global csv_out
    save_file = filedialog.asksaveasfilename(defaultextension =".csv", 
                                            filetypes = [("CSV files", "*.csv")], initialfile = csv_out)

    if save_file:
        content = pd.read_csv(csv_out)
        content.to_csv(save_file, index = False)
        messagebox.showinfo("Success", "File Saved")

# Main Tkinter app
app = tk.Tk()
app.title("Graphs")

#button loads a file
load_button = tk.Button(app, text="Load File", command = load_file)
load_button.pack()

#button saves file
save_button = tk.Button(app, text = "Save File", command = save_file)
save_button.pack()

app.mainloop()
