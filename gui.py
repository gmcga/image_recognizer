## gui.py ##
## Authors: 
## Description:
#       GUI file for Image Recognizer ML Software
#       © 2023, Graeme McGaughey and Kyle Sung


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw
from datetime import datetime
import image_rec as ir
import os


class ImageRec:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Recognizer")

        # Set window size
        WINDOW_WIDTH = 560
        WINDOW_HEIGHT = 560
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        
        # Explanation label
        self.introduction_label = tk.Label(root, text="Use your mouse to draw on the canvas!")
        self.introduction_label.pack()   

        # Create canvas:
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white", highlightbackground="black", highlightthickness=1)
        self.canvas.pack()

        # Bind LMB movement to canvas, call draw method:
        self.canvas.bind("<B1-Motion>", self.draw)

        # Button frame to hold the buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady= 10)

        # Clear button
        self.clear_button = tk.Button(button_frame, text="Clear Canvas", command=self.clear_canvas, width=15, height=3)
        self.clear_button.pack(side="left", padx=5)

        # Save button
        self.save_button = tk.Button(button_frame, text="Save Image", command=self.save_image, width=15, height=3)
        self.save_button.pack(side="left", padx=5)

        # Guess button
        self.guess_button = tk.Button(button_frame, text="Guess!", command=self.guess_image, width=15, height=3)
        self.guess_button.pack(side="left", padx=5)

        # Model's guess label
        self.guess_label = tk.Label(root, text="")
        self.guess_label.pack()

        # Authors and model label at bottom of window
        self.bottom_label = tk.Label(root, text=f"Running model {ir.get_model()} \n Made by Graeme McGaughey and Kyle Sung")
        self.bottom_label.pack(side="bottom", pady=10)
        # Initialize PIL image for drawing
        self.image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)


    def clear_canvas(self):
        self.canvas.delete("all") # Clear tkinter canvas
        self.image = Image.new("RGB", (280, 280), "white")  # Clear PIL image
        self.draw = ImageDraw.Draw(self.image)  # Create a new ImageDraw object

    def draw(self, event):
        RADIUS = 5 
        # Draw on tk image
        x, y = event.x, event.y
        self.canvas.create_oval(x + RADIUS, y + RADIUS, x - RADIUS, y - RADIUS, fill="black")

        # Draw on the PIL image
        pil_x0 = x - RADIUS
        pil_y0 = y - RADIUS
        pil_x1 = x + RADIUS
        pil_y1 = y + RADIUS
        self.draw.ellipse((pil_x0, pil_y0, pil_x1, pil_y1), fill="black")


    def save_image(self):
        # Ask user for file name and location to save
        file_path = tk.filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        print(file_path)
        if file_path:
            # Save the PIL image as a file
            self.image.save(file_path)

    def guess_image(self):
        try:
            os.makedirs("./fig_guess")
        except FileExistsError:
            pass

        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        file_name = f"./fig_guess/{current_time}.png"
        self.image.save(file_name)

        guess = ir.load_and_predict(file_name)
        self.guess_label.config(text=f"Model's Guess: {guess}")
        
        os.system(f"rm -rf fig_guess")


def main():

    root = tk.Tk()

    app = ImageRec(root)

    root.mainloop()

if __name__ == "__main__":
    main()
