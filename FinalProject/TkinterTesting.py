import tkinter as tk
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from tkinter import filedialog


#create and return a graph from a CSV file
def create_graph(csv_file):

    #need to add graph for 1/0 walking/jumping
    df = pd.read_csv(csv_file)
    fig, ax = plt.subplots()
    ax.plot(df["Time (s)"], df["Linear Acceleration z (m/s^2)"])
    ax.set_title("Graph from " + os.path.basename(csv_file))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linear Acceleration z (m/s^2)")
    ax.set_ylim([-75, 75])
    return fig



# Function to update the displayed graph
def load_file():
    global canvas, csv_out
    csv_in = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    
    #call classifier
    csv_out = csv_in

    if canvas:
        canvas.get_tk_widget().destroy()  # Destroy the previous canvas
    fig = create_graph(csv_out)
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
app.title("Toggle Graph")

canvas = None
csv_out = None

#button loads a file
load_button = tk.Button(app, text="Load File", command = load_file)
load_button.pack()

#button saves file
save_button = tk.Button(app, text = "Save File", command = save_file)
save_button.pack()

app.mainloop()

