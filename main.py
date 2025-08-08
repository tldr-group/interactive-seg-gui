import tkinter as tk

from data_model import DataModel
from GUI import App

# filenames = ("data/noisy_NMC_cracks.png",)
# filenames = ("data/nmc_noisy_stack.tif",)
# filenames = ("data/cells.jpg",)
filenames = None

if __name__ == "__main__":
    root = tk.Tk()
    root.title("")

    model = DataModel()
    app = App(root, model, initial_img_paths=filenames)
    app.grid()

    root.mainloop()
