## main.py 
## Authors: Kyle Sung and Graeme McGaughey
## Description: Main file for Image Recognizer ML Software



# IMPORTS

import image_rec as ir
import gui



# FUNCTION DEFINITIONS

def main():

    print()
    print(  f"ML Image Recognizer\n"
            f"Created by Graeme McGaughey and Kyle Sung\n"
            f"Running {ir.get_model()[7:-4]}\n\n"
            f"-- Starting GUI --"
    )
    
    gui.main() # initialize GUI

    print(f"-- Ending Program --")



# SCRIPT

if __name__ == "__main__":
    main()