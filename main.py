## main.py 
## Authors: Graeme McGaughey and Kyle Sung
## Description: Main file for Image Recognizer ML Software


# IMPORTS

import image_rec as ir
import gui


# FUNCTION DEFINITIONS

def main():
    print(f"Running model {ir.get_model()}")
    gui.main(do_train_model = False)


# SCRIPT

if __name__ == "__main__":
    main()