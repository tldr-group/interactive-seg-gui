import tkinter as tk
from PIL import Image

from data_model import DataModel
from GUI import App


filename = "data/test.png"

if __name__ == "__main__":
    root = tk.Tk()
    root.title("")

    model = DataModel()
    app = App(root, model)
    app.grid()

    root.mainloop()
