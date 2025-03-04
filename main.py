import tkinter as tk
from PIL import Image

from data_model import DataModel
from GUI import App


filename = "data/test.png"

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Deep Feature Segmentation")

    model = DataModel()
    app = App(root, model, Image.open(filename))
    app.grid()

    root.mainloop()
