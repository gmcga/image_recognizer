from PIL import Image, ImageDraw, ImageFont
import os
import random

# Set up the parameters
output_dir = "handdrawn_digits"
num_samples_per_digit = 69
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
        for itera in range(num_samples_per_digit):


            # Select a random font
            random_font_file = random.choice(font_files)

            bad_fonts = ['marlett.ttf', 'webdings.ttf', 'symbol.ttf', 'wingding.ttf', 'SegoeIcons.ttf', 'segmdl2.ttf',
                         'holomdl2.ttf', 'SansSerifCollection.ttf', 'CascadiaCode.ttf', 'CascadiaMono.ttf'
            ]

            while random_font_file in bad_fonts:
                random_font_file = random.choice(font_files) ## REPICK


            random_font_path = os.path.join(fonts_directory, random_font_file)

            print(digit, itera, str(random_font_file))


            canvas = Image.new("L", canvas_size, 255)  # White canvas
            draw = ImageDraw.Draw(canvas)
            digit_font = ImageFont.truetype(font = random_font_path, size = random.randrange(100, 170))

            # Random position for the digit
            x, y = random.randrange(15, 135), random.randrange(15, 125)

            # Generate and draw the digit on the canvas
            draw.text((x, y), str(digit), fill=0, font=digit_font)

            # Save the image
            filename = os.path.join(output_dir, f"agDotted_{digit}_{itera}.png")

            add_random_white_dots_to_canvas(canvas, 800, max_radius = 3)

            canvas.save(filename)



def add_random_white_dots_to_canvas(canvas, num_dots, max_radius):
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)
    
    for _ in range(num_dots):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        radius = random.randint(1, max_radius)
        
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if 0 <= x + i < width and 0 <= y + j < height:
                    if i ** 2 + j ** 2 <= radius ** 2:
                        draw.point((x + i, y + j), fill=255)  # Set pixel to white




if __name__ == "__main__":

    generate_and_save_digit_images()


