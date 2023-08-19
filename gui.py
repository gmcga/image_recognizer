import tkinter as tk

class ImageRec:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Recognizer")
        # Create canvas:
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        # Bind LMB movement to canvas, call draw method:
        self.canvas.bind("<B1-Motion>", self.draw)

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x+5, y+5, x-5, y-5, fill='black')


def main():

    root = tk.Tk()

    app = ImageRec(root)

    root.mainloop()

if __name__ == "__main__":
    main()
