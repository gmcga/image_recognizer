## main.py 
## Authors: Kyle Sung and Graeme McGaughey
## Description: Main file for Image Recognizer ML Software


# IMPORTS

import image_rec as ir
import gui


# FUNCTION DEFINITIONS

def main():
    print(f"Running {ir.get_model()}")
    print("--Starting GUI--")
    gui.main()


# SCRIPT

if __name__ == "__main__":
    main()