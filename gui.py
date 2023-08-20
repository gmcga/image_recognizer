## gui.py
## Authors: Graeme McGaughey and Kyle Sung
## Description: GUI file for Image Recognizer ML Software


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw
from datetime import datetime
import image_rec as ir
import torch
import torchvision as tv



MODEL_NUMBER = ir.get_model().split('_')[0].replace("models/model", "").replace(".pth", "")


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

        # Colour and erase buttons and frame to hold them
        colour_button_frame = tk.Frame(button_frame)
        colour_button_frame.pack(pady = 10)

        self.draw_colour = "black"

        self.erase_button = tk.Button(colour_button_frame, text="Eraser", command=self.toggle_eraser, width=5)
        self.erase_button.pack(side="top", anchor="ne")

        self.pen_button = tk.Button(colour_button_frame, text="Pen", command=self.toggle_pen, width=5)
        self.pen_button.pack(side="bottom", anchor="se")


        # Model's guess label
        self.guess_label = tk.Label(root, text="", font=20)
        self.guess_label.pack()

        # Authors and model label at bottom of window
        self.bottom_label = tk.Label(root, text=f"Running Image Recognition Model {MODEL_NUMBER} \n Made by Graeme McGaughey and Kyle Sung")
        self.bottom_label.pack(side="bottom", pady=10)

        # Initialize PIL image for drawing
        self.image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

    def toggle_eraser(self):
        self.draw_colour = "white"

    def toggle_pen(self):
        self.draw_colour = "black"


    def clear_canvas(self):
        self.canvas.delete("all") # Clear tkinter canvas
        self.image = Image.new("RGB", (280, 280), "white")  # Clear PIL image
        self.draw = ImageDraw.Draw(self.image)  # Create a new ImageDraw object


    def draw(self, event):
        RADIUS = 5 
        
        # Draw on the tk image canvas
        x, y = event.x, event.y
        self.canvas.create_oval(x + RADIUS, y + RADIUS, x - RADIUS, y - RADIUS, fill=self.draw_colour, outline=self.draw_colour)
        # Draw on the PIL image
        pil_x0 = x - RADIUS
        pil_y0 = y - RADIUS
        pil_x1 = x + RADIUS
        pil_y1 = y + RADIUS
        self.draw.ellipse((pil_x0, pil_y0, pil_x1, pil_y1), fill=self.draw_colour)

        # Automatically guess image after drawing:
        self.guess_image()


    def save_image(self):
        # Ask user for file name and location to save
        file_path = tk.filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        
        if file_path:
            # Save the PIL image as a file
            self.image.save(file_path)


    def guess_image(self):
        # Convert the current drawing on the canvas to a tensor
        pil_image = self.image.copy()
        pil_image = pil_image.resize((128, 128))  # Resize to match the model input size
        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        image_tensor = transform(pil_image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        net = ir.Net(10)  # 10 classes
        net.load_state_dict(torch.load(ir.get_model()))
        net.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            outputs = net(image_tensor)
            _, predicted = torch.max(outputs.data, 1)

        predicted_class = predicted.item()
        self.guess_label.config(text=f"Guessed Number: {predicted_class}")




def main():

    root = tk.Tk()

    
    app = ImageRec(root)

    root.mainloop()



if __name__ == "__main__":
    main()
