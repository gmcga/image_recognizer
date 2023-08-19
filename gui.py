import tkinter as tk

class ImageRec:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+10, y+10, fill='black')

root = tk.Tk()
print("yo")
app = ImageRec(root)
print("here")
root.mainloop()

print("hi")
