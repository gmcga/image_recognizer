from PIL import Image, ImageDraw, ImageFont
import os
import random

# Set up the parameters
output_dir = "handdrawn_digits"
num_samples_per_digit = 300
canvas_size = (280, 280)
pen_thickness = 2
digit_range = range(10)

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)



# Directory containing your TrueType fonts
fonts_directory = "C:\Windows\Fonts" #"c/Windows/Fonts/"

# Get a list of all TrueType fonts in the directory
font_files = [file for file in os.listdir(fonts_directory) if file.endswith('.ttf')]





# Function to draw digits and save images
def generate_and_save_digit_images():
    for digit in digit_range:
        for _ in range(num_samples_per_digit):


            # Select a random font
            random_font_file = random.choice(font_files)
            random_font_path = os.path.join(fonts_directory, random_font_file)



            canvas = Image.new("L", canvas_size, 255)  # White canvas
            draw = ImageDraw.Draw(canvas)
            digit_font = ImageFont.truetype(font = random_font_path, size = random.randrange(100, 150))

            # Random position for the digit
            x, y = random.randrange(15, 135), random.randrange(15, 135)

            # Generate and draw the digit on the canvas
            draw.text((x, y), str(digit), fill=0, font=digit_font)

            # Save the image
            filename = os.path.join(output_dir, f"autogen_{digit}_{_}.png")
            canvas.save(filename)

generate_and_save_digit_images()
