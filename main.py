import tkinter as tk

from data_model import DataModel
from GUI import App

filenames = None

if __name__ == "__main__":
    root = tk.Tk()
    root.title("")

    model = DataModel()
    app = App(root, model, initial_img_paths=filenames)
    app.grid()

    root.mainloop()
