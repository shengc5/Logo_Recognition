# Main.py
# TianYang Jin, Sheng, Chen
# CSE 415 Project
#


from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import NN_predict
import NB_predict

class App():
    def __init__(self, main):
        self.label = ttk.Label(main, text="Your car logo is: ")
        self.label.grid(column=3, row=1)

        self.canvas = Canvas(main, width=600, height=400)
        self.canvas.grid(row=5, column=3)

        white_img = np.ones((600, 400), dtype=np.uint8) * 255
        self.photo = ImageTk.PhotoImage(Image.fromarray(white_img))
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=NW, image=self.photo)
        self.button = Button(main, text="Choose Picture", command=self.open_image).grid(column=3, row=3)

    def open_image(self):
        img_path = filedialog.askopenfilename(filetypes=
                                              [("Image files", ("*.jpg", "*.jpeg", "*.bmp", "*.png"))])
        image = Image.open(img_path).resize((600, 400))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.image_on_canvas, image=self.photo)

        self.label.config(text=("Your car logo is: " + self.predict(img_path)))

    def predict(self, image_path):
        l = []
        l.append(NN_predict.predictImage(image_path, 'thetas'))
        l.append(NN_predict.predictImage(image_path, 'thetas'))
        # l.append(NN_predict.predictImage(image_path, 'thetas3'))
        # l.append(NN_predict.predictImage(image_path, 'thetas4'))
        # l.append(NN_predict.predictImage(image_path, 'thetas5'))
        l.append(NB_predict.predict(image_path, 'nb_data.npz'))
        return max(set(l), key=l.count)

root = Tk()
App(root)
root.mainloop()
