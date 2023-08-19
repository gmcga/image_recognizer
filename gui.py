import tkinter as tk

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

    def clear_canvas(self):
        self.canvas.delete("all")

    def draw(self, event): # Creates oval at cursor location
        x, y = event.x, event.y
        self.canvas.create_oval(x+5, y+5, x-5, y-5, fill="black")


def main():

    root = tk.Tk()

    app = ImageRec(root)

    root.mainloop()

if __name__ == "__main__":
    main()
