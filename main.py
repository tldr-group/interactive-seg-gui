import tkinter as tk
from PIL import Image

from data_model import DataModel
from GUI import App


filename = "data/005.tif"

if __name__ == "__main__":
    root = tk.Tk()
    root.title("")

    model = DataModel()
    app = App(root, model, filename)
    app.grid()

    root.mainloop()
