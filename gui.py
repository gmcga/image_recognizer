import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw


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

        # Clear button
        self.clear_button = tk.Button(root, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.pack()

        # Save button
        self.save_button = tk.Button(root, text="Save Image", command=self.save_image)
        self.save_button.pack()

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

        if file_path:
            # Save the PIL image as a file
            self.image.save(file_path)

def main():

    root = tk.Tk()

    app = ImageRec(root)

    root.mainloop()

if __name__ == "__main__":
    main()
