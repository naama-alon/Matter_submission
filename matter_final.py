import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import math
import os
import shutil

#--------------------------------------------------
#for typing text

def ensure_directory_exists(directory_path):
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # Create the directory
        try:
            os.makedirs(directory_path)
            print(f"Directory created: {directory_path}")
        except OSError as e:
            # Handle the error that occurred while creating the directory
            print(f"An error occurred while creating the directory: {e}")
    else:
        print("Directory already exists.")

def erase_files_in_directory(directory):
    # List directory contents
    try:
        dir_contents = os.listdir(directory)
    except FileNotFoundError:
        print("The specified directory does not exist.")
        return
    except PermissionError:
        print("You do not have permissions to access this directory.")
        return

    if not dir_contents:
        print("The directory is already empty.")
    else:
        # Loop through all files in the directory
        for item in dir_contents:
            file_path = os.path.join(directory, item)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Remove files and symbolic links
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directories
        print("All files and directories have been erased.")

def create_word_images(words):
    # Settings for image size and font
    image_size = (500, 500)
    font_size = 130
    font_path = "C:\\Windows\\Fonts\\FRAHV.ttf"  # font Franklin Gothic Heavy
    dir = f"images/typed_text/"
    ensure_directory_exists(dir)
    erase_files_in_directory(dir)
    
    for i, word in enumerate(words):
        # Create a blank image with white background
        image = Image.new("RGB", image_size, "white")
        draw = ImageDraw.Draw(image)

        # Load the font
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"Font file not found: {font_path}")
            return

        # Calculate text width and height
        text_width, text_height = draw.textsize(word, font=font)

        # Calculate position to center the text
        x = (image_size[0] - text_width) / 2
        y = (image_size[1] - text_height) / 2

        # Draw the text on the image
        draw.text((x, y), word, font=font, fill="black")

        # Save the image
        image.save(dir + f"word_{i + 1}.png")

def validate_input(input_words):
    if not input_words:
        print("Please enter at least one word.")
        return False
    if len(input_words) > 4:
        print("You typed more than 4 words. Please enter up to 4 words.")
        return False
    if any(len(word) > 7 for word in input_words):
        print("Please type words with a maximum of 7 letters.")
        return False
    return True

def get_valid_input():
    while True:
        # Input words from the user
        input_words = input("Enter up to 4 words, each up to 7 letters, separated by spaces: ").split()
        
        # Validate input
        if validate_input(input_words):
            return input_words
        else:
            print("Please try again.")

def type_text():
    # Get valid input from user
    valid_words = get_valid_input()

    # Create images with valid input
    create_word_images(valid_words)

#--------------------------------------------------
# The Algorithm:

"""
Degrees explanation:
angle_degrees - specifies the angle in degrees for the displacement direction, where:
* 0 degrees is to the right (positive x-direction).
* 90 degrees is downward (positive y-direction).
* 180 degrees is to the left (negative x-direction).
* 270 degrees (or -90 degrees) is upward (negative y-direction).
move_dist_mm - specifies how far from the origin (central position) the center of the circle should be placed, following the direction specified by angle_degrees.

1 Center Circle:
move_dist_mm = 0, angle_degrees = 0 (No displacement)

2 Right Circle:
move_dist_mm = some_positive_value, angle_degrees = 0 (Move right)

3 Left Circle:
move_dist_mm = some_positive_value, angle_degrees = 180 (Move left)

4 Up Circle:
move_dist_mm = some_positive_value, angle_degrees = 270 (Move up)

5 Down Circle:
move_dist_mm = some_positive_value, angle_degrees = 90 (Move down)
"""

def count_files_in_directory(directory):
    # List directory contents
    try:
        dir_contents = os.listdir(directory)
    except FileNotFoundError:
        print("The specified directory does not exist.")
        return
    except PermissionError:
        print("You do not have permissions to access this directory.")
        return

    # Count files (ignoring directories)
    file_count = len([file for file in dir_contents if os.path.isfile(os.path.join(directory, file))])
    
    # Print the file count
    print(f"There are {file_count} files in the directory.")
    return file_count

def mm_to_pixels(mm, dpi):
    #Convert millimeters to pixels as a float for precision.
    return mm * dpi / 25.4

def calculate_rows_columns(size_mm, dist_between_circles_in_row, vertical_dist):
    #Calculate the maximum number of rows and columns that fit within the specified size in mm.
    #size_px = mm_to_pixels(size_mm, dpi)
    num_columns = math.floor(size_mm // dist_between_circles_in_row)
    num_rows = math.floor(size_mm // vertical_dist)
    #return math.ceil(num_rows/2), num_columns
    return num_rows, num_columns

def create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, move_dist_mm, angle_degrees):
    #Create a mask based on specified parameters with floating-point precision.
    rows, columns = calculate_rows_columns(size_mm, dist_between_circles_in_row, vertical_dist)
    circle_radius_px = mm_to_pixels(circle_radius, dpi)
    shift_odd_rows_px = mm_to_pixels(shift_odd_rows, dpi)
    dist_between_circles_in_row_px = mm_to_pixels(dist_between_circles_in_row, dpi)
    vertical_dist_px = mm_to_pixels(vertical_dist, dpi)
    move_dist_px = mm_to_pixels(move_dist_mm, dpi)

    angle_radians = math.radians(angle_degrees)
    move_x = move_dist_px * math.cos(angle_radians)
    move_y = move_dist_px * math.sin(angle_radians)

    print("Moving distance (px):", move_dist_px)
    print("Angle (degrees):", angle_degrees)
    print("Calculated radians:", angle_radians)
    print("Move X:", move_x)
    print("Move Y:", move_y)

    width = int(round((columns - 1) * dist_between_circles_in_row_px +
                       2 * circle_radius_px + abs(move_x)))
    height = int(round((rows - 1) * vertical_dist_px + 2 * circle_radius_px + abs(move_y)))

    mask = np.ones((height, width), dtype=np.uint8) * 255  # White background

    for row in range(rows):
        y_pos = row * vertical_dist_px + move_y + circle_radius_px
        for col in range(columns):
            x_pos = col * dist_between_circles_in_row_px + move_x + (shift_odd_rows_px if row % 2 else 0) + circle_radius_px
            for y in np.arange(-circle_radius_px, circle_radius_px + 1, 0.5):
                for x in np.arange(-circle_radius_px, circle_radius_px + 1, 0.5):
                    if (x + 0.5)**2 + (y + 0.5)**2 < circle_radius_px**2:
                        yy = int(round(y_pos + y))
                        xx = int(round(x_pos + x))
                        if 0 <= yy < height and 0 <= xx < width:
                            mask[yy, xx] = 0  # Black circle
    return mask

"""
def create_circle_on_elliptical_base_mm(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, move_dist_mm, angle_degrees):
    #Create a mask based on specified parameters with floating-point precision.
    rows, columns = calculate_rows_columns(size_mm, dist_between_circles_in_row, vertical_dist)
    circle_radius_px = mm_to_pixels(circle_radius, dpi)
    shift_odd_rows_px = mm_to_pixels(shift_odd_rows, dpi)
    dist_between_circles_in_row_px = mm_to_pixels(dist_between_circles_in_row, dpi)
    vertical_dist_px = mm_to_pixels(vertical_dist, dpi)
    move_dist_px = mm_to_pixels(move_dist_mm, dpi)

    angle_radians = math.radians(angle_degrees)
    move_x = move_dist_mm * math.cos(angle_radians)
    move_y = move_dist_mm * math.sin(angle_radians)

    width = int(round(mm_to_pixels((columns - 1) * dist_between_circles_in_row + 2 * circle_radius + abs(move_x),dpi)))
    height = int(round(mm_to_pixels((rows - 1) * vertical_dist + 2 * circle_radius + abs(move_y),dpi)))

    mask = np.ones((height, width), dtype=np.uint8) * 255  # White background

    for row in range(rows):
        y_pos = row * vertical_dist + move_y + circle_radius
        for col in range(columns):
            x_pos = col * dist_between_circles_in_row + move_x + (shift_odd_rows if row % 2 else 0) + circle_radius
            for y in np.arange(-circle_radius, circle_radius + 1, 0.5):
                for x in np.arange(-circle_radius, circle_radius + 1, 0.5):
                    if (x + 0.5)**2 + (y + 0.5)**2 < circle_radius**2:
                        yy = int(round(mm_to_pixels(y_pos + y,dpi)))
                        xx = int(round(mm_to_pixels(x_pos + x,dpi)))
                        if 0 <= yy < height and 0 <= xx < width:
                            mask[yy, xx] = 0  # Black circle
    return mask
"""
def create_square_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius, move_dist_mm, angle_degrees):
    # Calculate rows and columns
    rows, columns = calculate_rows_columns(size_mm, dist_between_circles_in_row, vertical_dist)
    square_side_px = mm_to_pixels(2 * square_radius, dpi)  # Full side length of the square is 2 times the radius
    shift_odd_rows_px = mm_to_pixels(shift_odd_rows, dpi)
    dist_between_circles_in_row_px = mm_to_pixels(dist_between_circles_in_row, dpi)
    vertical_dist_px = mm_to_pixels(vertical_dist, dpi)
    move_dist_px = mm_to_pixels(move_dist_mm, dpi)

    angle_radians = math.radians(angle_degrees)
    move_x = move_dist_px * math.cos(angle_radians)
    move_y = move_dist_px * math.sin(angle_radians)

    width = int(round((columns - 1) * dist_between_circles_in_row_px + square_side_px + abs(move_x)))
    height = int(round((rows - 1) * vertical_dist_px + square_side_px + abs(move_y)))

    mask = np.ones((height, width), dtype=np.uint8) * 255  # White background

    # Calculate the half width of the square for easier bounds calculation
    half_square_side_px = square_side_px / 2

    for row in range(rows):
        y_pos = row * vertical_dist_px + move_y + vertical_dist_px / 2
        for col in range(columns):
            x_pos = col * dist_between_circles_in_row_px + move_x + (shift_odd_rows_px if row % 2 else 0) + dist_between_circles_in_row_px / 2
            top_left_x = int(x_pos - half_square_side_px)
            top_left_y = int(y_pos - half_square_side_px)
            bottom_right_x = int(x_pos + half_square_side_px)
            bottom_right_y = int(y_pos + half_square_side_px)

            # Fill in the square
            for y in range(max(0, top_left_y), min(height, bottom_right_y)):
                for x in range(max(0, top_left_x), min(width, bottom_right_x)):
                    mask[y, x] = 0  # Set pixel to black

    return mask

def apply_mask_to_image(image_path, mask, save_path, target_dpi=300):
    #Apply a generated mask to an image and resize the image to fit the mask without changing its proportions.
    image = Image.open(image_path)
    original_width, original_height = image.size
    mask_width, mask_height = mask.shape[1], mask.shape[0]

    # Calculate the scaling factor to ensure the image fits within the mask dimensions
    scale_width = mask_width / original_width
    scale_height = mask_height / original_height
    scale_factor = min(scale_width, scale_height)

    # Resize image using the smallest scaling factor to maintain aspect ratio
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    image = image.resize((new_width, new_height), Image.ANTIALIAS)
    #image.save('image_resized.png',dpi=(dpi, dpi))
    image.save('image_resized.jpg',dpi=(target_dpi, target_dpi))

    """
    plt.imshow(image, cmap='gray')
    plt.title("Resized Image")
    plt.axis('off')
    plt.show()
    """

    # Center the image on the mask
    offset_x = (mask_width - new_width) // 2
    offset_y = (mask_height - new_height) // 2
    new_image = Image.new("RGB", (mask_width, mask_height), "white")
    new_image.paste(image, (offset_x, offset_y))

    #plt.figure(figsize=(10, 10))
    """
    plt.figure()
    plt.imshow(new_image)
    plt.title("Image Fitted to Mask")
    plt.axis('off')
    plt.show()
    """

    # Apply mask
    new_image_array = np.array(new_image)
    mask_array = np.where(mask > 0, 1, 0)  # Invert mask for use with images
    masked_image_array = np.where(mask_array[..., None]==0, new_image_array, [255, 255, 255])
    masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
    masked_image.save(save_path, dpi=(target_dpi, target_dpi))  # Save with specified DPI

    """
    plt.imshow(masked_image, cmap='gray')
    plt.title("Masked Image")
    plt.axis('off')
    plt.show()
    """
    return masked_image_array

def combine_masks(mask1, mask2):
    height_diff = mask2.shape[0] - mask1.shape[0]
    width_diff = mask2.shape[1] - mask1.shape[1]

    mask1_padded = mask1
    mask2_padded = mask2
        
    if height_diff > 0 or width_diff > 0:
        pad_top = max(height_diff // 2, 0)
        pad_bottom = max(height_diff - pad_top, 0)
        pad_left = max(width_diff // 2, 0)
        pad_right = max(width_diff - pad_left, 0)
        mask1_padded = np.pad(mask1, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=255)
        
    if height_diff < 0 or width_diff < 0:
        pad_top = max(-height_diff // 2, 0)
        pad_bottom = max(-height_diff - pad_top, 0)
        pad_left = max(-width_diff // 2, 0)
        pad_right = max(-width_diff - pad_left, 0)
        mask2_padded = np.pad(mask2, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=255)

    combined_mask = np.logical_and(mask1_padded == 255, mask2_padded == 255) * 255
    return combined_mask, mask1_padded, mask2_padded


#circle
def one_image(image_path, dpi,size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius):

    mask = create_circle_on_elliptical_base(dpi,size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0, 0)
    
    #plt.figure(figsize=(10, 10))
    """
    plt.imshow(mask, cmap='gray', interpolation='none')
    plt.title("Combined Circular Mask")
    plt.axis('off')
    plt.show()
    """

    masked_image = apply_mask_to_image(
    image_path=image_path,
    mask=mask,
    save_path='masked_image.png',
    target_dpi=dpi
    )

    image = Image.fromarray(masked_image.astype(np.uint8))
    image.save('masked_image.png',dpi=(dpi, dpi))
    image.show()

def two_images(image_path_1, image_path_2, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius):

    mask1 = create_circle_on_elliptical_base(dpi,size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0, 0)
    mask2 = create_circle_on_elliptical_base(dpi,size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 180)

    combined_mask, mask1_padded, mask2_padded = combine_masks(mask1, mask2)

    #plt.figure(figsize=(10, 10))
    """
    plt.imshow(combined_mask, cmap='gray', interpolation='none')
    plt.title("Combined Circular Mask")
    plt.axis('off')
    plt.show()
    """
    #-----
    # Convert masks to boolean arrays (where 0 means inside the circle, 255 means background)
    mask1_boolean = mask1_padded == 0  # Black circles in mask1
    mask2_boolean = mask2_padded == 0  # Black circles in mask2

    # Create an empty RGB image with a white background (3 channels - R, G, B)
    height, width = mask1_padded.shape
    colored_mask = np.ones((height, width, 3))  # Initialize with all white (RGB: [1, 1, 1])

    # Set red for mask1
    colored_mask[mask1_boolean] = [1, 0, 0]  # Red color for mask1

    # Set green for mask2 where mask1 is not applied
    colored_mask[mask2_boolean & ~mask1_boolean] = [0, 1, 0]  # Green color for mask2

    # Show the combined mask with different colors on a white background
    #plt.figure(figsize=(10, 10))
    """
    plt.imshow(colored_mask, interpolation='none')
    plt.title("White Background with Red (Mask1), Green (Mask2)")
    plt.axis('off')  # Turn off axis labels
    plt.show()
    """

    # Convert the colored mask (values in [0, 1]) to [0, 255] for saving
    colored_mask_uint8 = (colored_mask * 255).astype(np.uint8)
    # Convert the mask to an image and save it
    img = Image.fromarray(colored_mask_uint8)
    img.save('combined_colored_masks.png',dpi=(dpi, dpi))
    #-----

    masked_image1 = apply_mask_to_image(
        image_path=image_path_1,
        mask=mask1_padded,
        save_path='masked_image_1.jpg',
        target_dpi=dpi
    )

    masked_image2 = apply_mask_to_image(
        image_path=image_path_2,
        mask=mask2_padded,
        save_path='masked_image_2.jpg',
        target_dpi=dpi
    )


    # Combine images by taking the maximum value for each pixel
    combined_image_array = np.minimum(masked_image1, masked_image2)
    # Convert back to a PIL Image
    combined_image = Image.fromarray(combined_image_array.astype(np.uint8))

    #combined_image.save('combined_masked_image.jpg',dpi=(dpi, dpi))
    combined_image.save('combined_masked_image.png',dpi=(dpi, dpi))
    combined_image.show()


#circle experiments
def three_images(image_path_1, image_path_2, image_path_3,dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius):

    #mask2 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0, 0) # middle
    #mask1 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, -1.5, 270) #left
    #mask3 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 270) #right

    mask2 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0, 0) # center
    mask3 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.5, 0) # right
    mask1 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 180) #left

    _, mask1_padded, mask2_padded = combine_masks(mask1, mask2)
    combined_mask_tmp, mask1_padded, mask3_padded = combine_masks(mask1_padded, mask3)
    combined_mask, _, mask2_padded = combine_masks(combined_mask_tmp, mask2_padded)

    """
    #plt.figure(figsize=(10, 10))
    plt.imshow(combined_mask, cmap='gray', interpolation='none')
    plt.title("Combined Circular Mask")
    plt.axis('off')
    plt.show()
    """

        #-----
    # Convert masks to boolean arrays (where 0 means inside the circle, 255 means background)
    mask1_boolean = mask1_padded == 0  # Black circles in mask1
    mask2_boolean = mask2_padded == 0  # Black circles in mask2
    mask3_boolean = mask3_padded == 0  # Black circles in mask3

    # Create an empty RGB image with a white background (3 channels - R, G, B)
    height, width = mask1_padded.shape
    colored_mask = np.ones((height, width, 3))  # Initialize with all white (RGB: [1, 1, 1])

    # Set red for mask1
    colored_mask[mask1_boolean] = [1, 0, 0]  # Red color for mask1

    # Set green for mask2 where mask1 is not applied
    colored_mask[mask2_boolean & ~mask1_boolean] = [0, 1, 0]  # Green color for mask2

    # Set blue for mask3 where mask1 and mask2 are not applied
    colored_mask[mask3_boolean & ~mask1_boolean & ~mask2_boolean] = [0, 0, 1]  # Blue color for mask3

    # Show the combined mask with different colors on a white background
    #plt.figure(figsize=(10, 10))
    """
    plt.figure()
    plt.imshow(colored_mask, interpolation='none')
    plt.title("White Background with Yellow (Mask1), Red (Mask2), Blue (Mask3)")
    plt.axis('off')  # Turn off axis labels
    plt.show()
    """

    # Convert the colored mask (values in [0, 1]) to [0, 255] for saving
    colored_mask_uint8 = (colored_mask * 255).astype(np.uint8)
    # Convert the mask to an image and save it
    img = Image.fromarray(colored_mask_uint8)
    img.save('combined_colored_masks.png',dpi=(dpi, dpi))
    #-----

    masked_image1 = apply_mask_to_image(
        image_path=image_path_1,
        mask=mask1_padded,
        save_path='masked_image_1.jpg',
        target_dpi=dpi
    )

    masked_image2 = apply_mask_to_image(
        image_path=image_path_2,
        mask=mask2_padded,
        save_path='masked_image_2.jpg',
        target_dpi=dpi
    )

    masked_image3 = apply_mask_to_image(
        image_path=image_path_3,
        mask=mask3_padded,
        save_path='masked_image_3.jpg',
        target_dpi=dpi
    )


    # Combine images by taking the maximum value for each pixel
    combined_image_array_tmp = np.minimum(masked_image1, masked_image2)
    combined_image_array = np.minimum(combined_image_array_tmp, masked_image3)
    # Convert back to a PIL Image
    combined_image = Image.fromarray(combined_image_array.astype(np.uint8))

    #combined_image.save('combined_masked_image.jpg',dpi=(dpi, dpi))
    combined_image.save('combined_masked_image.png',dpi=(dpi, dpi))
    combined_image.show()

def four_images(image_path_1, image_path_2, image_path_3,image_path_4, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius):

    #mask1 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0, 0)
    #mask2 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 180)
    #mask3 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 270)
    #mask4 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, -1.5, 270)
    """
    mask1 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.5, 0) #right
    mask2 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 180) #left
    mask3 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 270) #up
    mask4 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.5, 90) #down
    """
    mask1 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.5, 0) #right
    mask2 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 180) #left
    mask3 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 270) #up
    mask4 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.5, 90) #down



    combined_mask_tmp1, mask1_padded, mask4_padded = combine_masks(mask1, mask4)
    combined_mask_tmp2, _, mask2_padded = combine_masks(combined_mask_tmp1, mask2)
    combined_mask, _, mask3_padded = combine_masks(combined_mask_tmp2, mask3)
    _, _, mask1_padded = combine_masks(combined_mask, mask1)
    _, _, mask2_padded = combine_masks(combined_mask, mask2)
    _, _, mask4_padded = combine_masks(combined_mask, mask4)


    #plt.figure(figsize=(10, 10))
    """
    plt.figure()
    plt.imshow(combined_mask, cmap='gray', interpolation='none')
    plt.title("Combined Circular Mask")
    plt.axis('off')
    plt.show()
    """
    #black_mask_uint8 = (combined_mask * 255).astype(np.uint8)
    #img_black = Image.fromarray(black_mask_uint8)
    #img_black.save('combined_mask_black.png',dpi=(dpi, dpi))
    #-----
    # Convert masks to boolean arrays (where 0 means inside the circle, 255 means background)
    mask1_boolean = mask1_padded == 0  # Black circles in mask1
    mask2_boolean = mask2_padded == 0  # Black circles in mask2
    mask3_boolean = mask3_padded == 0  # Black circles in mask3
    mask4_boolean = mask4_padded == 0  # Black circles in mask4

    # Create an empty RGB image with a white background (3 channels - R, G, B)
    height, width = mask1_padded.shape
    colored_mask = np.ones((height, width, 3))  # Initialize with all white (RGB: [1, 1, 1])

    # Set red for mask1
    colored_mask[mask1_boolean] = [1, 0, 0]  # Red color for mask1

    # Set green for mask2 where mask1 is not applied
    colored_mask[mask2_boolean & ~mask1_boolean] = [0, 1, 0]  # Green color for mask2

    # Set blue for mask3 where mask1 and mask2 are not applied
    colored_mask[mask3_boolean & ~mask1_boolean & ~mask2_boolean] = [0, 0, 1]  # Blue color for mask3

    # Set yellow for mask4 where mask1, mask2, and mask3 are not applied
    colored_mask[mask4_boolean & ~mask1_boolean & ~mask2_boolean & ~mask3_boolean] = [1, 1, 0]  # Yellow color for mask4

    # Show the combined mask with different colors on a white background
    #plt.figure(figsize=(10, 10))
    """
    plt.figure()
    plt.imshow(colored_mask, interpolation='none')
    plt.title("White Background with Red (Mask1), Green (Mask2), Blue (Mask3), and Yellow (Mask4)")
    plt.axis('off')  # Turn off axis labels
    plt.show()
    """

    # Convert the colored mask (values in [0, 1]) to [0, 255] for saving
    colored_mask_uint8 = (colored_mask * 255).astype(np.uint8)
    # Convert the mask to an image and save it
    img = Image.fromarray(colored_mask_uint8)
    img.save('combined_colored_masks.png',dpi=(dpi, dpi))
    #-----

    masked_image1 = apply_mask_to_image(
        image_path=image_path_1,
        mask=mask1_padded,
        save_path='masked_image_1.jpg',
        target_dpi=dpi
    )

    masked_image2 = apply_mask_to_image(
        image_path=image_path_2,
        mask=mask2_padded,
        save_path='masked_image_2.jpg',
        target_dpi=dpi
    )

    masked_image3 = apply_mask_to_image(
        image_path=image_path_3,
        mask=mask3_padded,
        save_path='masked_image_3.jpg',
        target_dpi=dpi
    )

    masked_image4 = apply_mask_to_image(
        image_path=image_path_4,
        mask=mask4_padded,
        save_path='masked_image_4.jpg',
        target_dpi=dpi
    )


    # Combine images by taking the maximum value for each pixel
    combined_image_array_tmp1 = np.minimum(masked_image1, masked_image2)
    combined_image_array_tmp2 = np.minimum(combined_image_array_tmp1, masked_image3)
    combined_image_array = np.minimum(combined_image_array_tmp2, masked_image4)
    # Convert back to a PIL Image
    combined_image = Image.fromarray(combined_image_array.astype(np.uint8))

    #combined_image.save('combined_masked_image.jpg',dpi=(dpi, dpi))
    combined_image.save('combined_masked_image.png',dpi=(dpi, dpi))
    combined_image.show()

def five_images(image_path_1, image_path_2, image_path_3,image_path_4, image_path_5, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius):

    """
    1 Center Circle:
    move_dist_mm = 0, angle_degrees = 0 (No displacement)

    2 Right Circle:
    move_dist_mm = some_positive_value, angle_degrees = 0 (Move right)

    3 Left Circle:
    move_dist_mm = some_positive_value, angle_degrees = 180 (Move left)

    4 Up Circle:
    move_dist_mm = some_positive_value, angle_degrees = 270 (Move up)

    5 Down Circle:
    move_dist_mm = some_positive_value, angle_degrees = 90 (Move down)
    """

    
    mask1 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0, 0) # center
    mask2 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.5, 0) #right
    mask3 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 180) #left
    mask4 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 270) #up
    mask5 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.5, 90) #down
    
    #mask1 = create_circle_on_elliptical_base_mm(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0, 0)
    #mask2 = create_circle_on_elliptical_base_mm(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.5, 0)
    #mask3 = create_circle_on_elliptical_base_mm(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 180)
    #mask4 = create_circle_on_elliptical_base_mm(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 270)
    #mask5 = create_circle_on_elliptical_base_mm(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.5, 90)



    combined_mask_tmp1, mask1_padded, mask4_padded = combine_masks(mask1, mask4)
    combined_mask_tmp2, _, mask2_padded = combine_masks(combined_mask_tmp1, mask2)
    combined_mask_tmp3, _, mask5_padded = combine_masks(combined_mask_tmp2, mask5)
    combined_mask, _, mask3_padded = combine_masks(combined_mask_tmp3, mask3)
    _, _, mask1_padded = combine_masks(combined_mask, mask1)
    _, _, mask2_padded = combine_masks(combined_mask, mask2)
    _, _, mask4_padded = combine_masks(combined_mask, mask4)
    _, _, mask5_padded = combine_masks(combined_mask, mask5)


    """
    plt.figure(figsize=(10, 10))
    plt.imshow(combined_mask, cmap='gray', interpolation='none')
    plt.title("Combined Circular Mask")
    plt.axis('off')
    plt.show()
    """
    #-----
    # Convert masks to boolean arrays (where 0 means inside the circle, 255 means background)
    mask1_boolean = mask1_padded == 0  # Black circles in mask1
    mask2_boolean = mask2_padded == 0  # Black circles in mask2
    mask3_boolean = mask3_padded == 0  # Black circles in mask3
    mask4_boolean = mask4_padded == 0  # Black circles in mask4
    mask5_boolean = mask5_padded == 0  # Black circles in mask5

    # Create an empty RGB image with a white background (3 channels - R, G, B)
    height, width = mask1_padded.shape
    colored_mask = np.ones((height, width, 3))  # Initialize with all white (RGB: [1, 1, 1])

    # Set red for mask1
    colored_mask[mask1_boolean] = [1, 0, 0]  # Red color for mask1

    # Set green for mask2 where mask1 is not applied
    colored_mask[mask2_boolean & ~mask1_boolean] = [0, 1, 0]  # Green color for mask2

    # Set blue for mask3 where mask1 and mask2 are not applied
    colored_mask[mask3_boolean & ~mask1_boolean & ~mask2_boolean] = [0, 0, 1]  # Blue color for mask3

    # Set yellow for mask4 where mask1, mask2, and mask3 are not applied
    colored_mask[mask4_boolean & ~mask1_boolean & ~mask2_boolean & ~mask3_boolean] = [1, 1, 0]  # Yellow color for mask4

    # Set magenta for mask5 where masks 1, 2, 3, and 4 are not applied
    colored_mask[mask5_boolean & ~mask1_boolean & ~mask2_boolean & ~mask3_boolean & ~mask4_boolean] = [1, 0, 1]  # Magenta color for mask5

    # Show the combined mask with different colors on a white background
    """
    #plt.figure(figsize=(10, 10))
    plt.imshow(colored_mask, interpolation='none')
    plt.title("White Background with Red (Mask1), Green (Mask2), Blue (Mask3), Yellow (Mask4) amd Magenta (Mask5)")
    plt.axis('off')  # Turn off axis labels
    plt.show()
    """

    # Convert the colored mask (values in [0, 1]) to [0, 255] for saving
    colored_mask_uint8 = (colored_mask * 255).astype(np.uint8)
    # Convert the mask to an image and save it
    img = Image.fromarray(colored_mask_uint8)
    img.save('combined_colored_masks.png',dpi=(dpi, dpi))
    #-----

    masked_image1 = apply_mask_to_image(
        image_path=image_path_1,
        mask=mask1_padded,
        save_path='masked_image_1.jpg',
        target_dpi=dpi
    )

    masked_image2 = apply_mask_to_image(
        image_path=image_path_2,
        mask=mask2_padded,
        save_path='masked_image_2.jpg',
        target_dpi=dpi
    )

    masked_image3 = apply_mask_to_image(
        image_path=image_path_3,
        mask=mask3_padded,
        save_path='masked_image_3.jpg',
        target_dpi=dpi
    )

    masked_image4 = apply_mask_to_image(
        image_path=image_path_4,
        mask=mask4_padded,
        save_path='masked_image_4.jpg',
        target_dpi=dpi
    )

    masked_image5 = apply_mask_to_image(
        image_path=image_path_5,
        mask=mask5_padded,
        save_path='masked_image_5.jpg',
        target_dpi=dpi
    )


    # Combine images by taking the maximum value for each pixel
    combined_image_array_tmp1 = np.minimum(masked_image1, masked_image2)
    combined_image_array_tmp2 = np.minimum(combined_image_array_tmp1, masked_image3)
    combined_image_array_tmp3 = np.minimum(combined_image_array_tmp2, masked_image4)
    combined_image_array = np.minimum(combined_image_array_tmp3, masked_image5)
    # Convert back to a PIL Image
    combined_image = Image.fromarray(combined_image_array.astype(np.uint8))

    #combined_image.save('combined_masked_image.jpg',dpi=(dpi, dpi))
    combined_image.save('combined_masked_image.png',dpi=(dpi, dpi))
    combined_image.show()

def moving_right_4(image_path_1, image_path_2, image_path_3,image_path_4, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius):
   
    mask1 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0, 0) #middle red
    mask2 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.8, 0) #right1 green
    mask3 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 3.6, 0) #right2 blue 
    mask4 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 5.4, 0) #right3 yellow
    
    """
    test 1 - 1.6, test 2 -1.7, test 3 - 1.8
    mask1 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0, 0) #middle red
    mask2 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.6, 0) #right1 green
    mask3 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 3.2, 0) #right2 blue 
    mask4 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 4.8, 0) #right3 yellow
    """


    combined_mask_tmp1, mask1_padded, mask4_padded = combine_masks(mask1, mask4)
    combined_mask_tmp2, _, mask2_padded = combine_masks(combined_mask_tmp1, mask2)
    combined_mask, _, mask3_padded = combine_masks(combined_mask_tmp2, mask3)
    _, _, mask1_padded = combine_masks(combined_mask, mask1)
    _, _, mask2_padded = combine_masks(combined_mask, mask2)
    _, _, mask4_padded = combine_masks(combined_mask, mask4)

    """
    plt.figure(figsize=(10, 10))
    plt.imshow(combined_mask, cmap='gray', interpolation='none')
    plt.title("Combined Circular Mask")
    plt.axis('off')
    plt.show()
    """
    #-----
    # Convert masks to boolean arrays (where 0 means inside the circle, 255 means background)
    mask1_boolean = mask1_padded == 0  # Black circles in mask1
    mask2_boolean = mask2_padded == 0  # Black circles in mask2
    mask3_boolean = mask3_padded == 0  # Black circles in mask3
    mask4_boolean = mask4_padded == 0  # Black circles in mask4

    # Create an empty RGB image with a white background (3 channels - R, G, B)
    height, width = mask1_padded.shape
    colored_mask = np.ones((height, width, 3))  # Initialize with all white (RGB: [1, 1, 1])

    # Set red for mask1
    colored_mask[mask1_boolean] = [1, 0, 0]  # Red color for mask1

    # Set green for mask2 where mask1 is not applied
    colored_mask[mask2_boolean & ~mask1_boolean] = [0, 1, 0]  # Green color for mask2

    # Set blue for mask3 where mask1 and mask2 are not applied
    colored_mask[mask3_boolean & ~mask1_boolean & ~mask2_boolean] = [0, 0, 1]  # Blue color for mask3

    # Set yellow for mask4 where mask1, mask2, and mask3 are not applied
    colored_mask[mask4_boolean & ~mask1_boolean & ~mask2_boolean & ~mask3_boolean] = [1, 1, 0]  # Yellow color for mask4

    # Show the combined mask with different colors on a white background
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(colored_mask, interpolation='none')
    plt.title("White Background with Red (Mask1), Green (Mask2), Blue (Mask3), and Yellow (Mask4)")
    plt.axis('off')  # Turn off axis labels
    plt.show()
    """

    # Convert the colored mask (values in [0, 1]) to [0, 255] for saving
    colored_mask_uint8 = (colored_mask * 255).astype(np.uint8)
    # Convert the mask to an image and save it
    img = Image.fromarray(colored_mask_uint8)
    img.save('combined_colored_masks.png',dpi=(dpi, dpi))
    #-----

    masked_image1 = apply_mask_to_image(
        image_path=image_path_1,
        mask=mask1_padded,
        save_path='masked_image_1.jpg',
        target_dpi=dpi
    )

    masked_image2 = apply_mask_to_image(
        image_path=image_path_2,
        mask=mask2_padded,
        save_path='masked_image_2.jpg',
        target_dpi=dpi
    )

    masked_image3 = apply_mask_to_image(
        image_path=image_path_3,
        mask=mask3_padded,
        save_path='masked_image_3.jpg',
        target_dpi=dpi
    )

    masked_image4 = apply_mask_to_image(
        image_path=image_path_4,
        mask=mask4_padded,
        save_path='masked_image_4.jpg',
        target_dpi=dpi
    )


    # Combine images by taking the maximum value for each pixel
    combined_image_array_tmp1 = np.minimum(masked_image1, masked_image2)
    combined_image_array_tmp2 = np.minimum(combined_image_array_tmp1, masked_image3)
    combined_image_array = np.minimum(combined_image_array_tmp2, masked_image4)
    # Convert back to a PIL Image
    combined_image = Image.fromarray(combined_image_array.astype(np.uint8))

    #combined_image.save('combined_masked_image.jpg',dpi=(dpi, dpi))
    combined_image.save('combined_masked_image.png',dpi=(dpi, dpi))
    combined_image.show()

def moving_right_3(image_path_1, image_path_2, image_path_3, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius):
   
    mask1 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0, 0) #middle red
    mask2 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.8, 0) #right1 green
    mask3 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 3.6, 0) #right2 blue 
    
    """
    """
    _, mask1_padded, mask2_padded = combine_masks(mask1, mask2)
    combined_mask_tmp, mask1_padded, mask3_padded = combine_masks(mask1_padded, mask3)
    combined_mask, _, mask2_padded = combine_masks(combined_mask_tmp, mask2_padded)

    """
    plt.figure(figsize=(10, 10))
    plt.imshow(combined_mask, cmap='gray', interpolation='none')
    plt.title("Combined Circular Mask")
    plt.axis('off')
    plt.show()
    """

        #-----
    # Convert masks to boolean arrays (where 0 means inside the circle, 255 means background)
    mask1_boolean = mask1_padded == 0  # Black circles in mask1
    mask2_boolean = mask2_padded == 0  # Black circles in mask2
    mask3_boolean = mask3_padded == 0  # Black circles in mask3

    # Create an empty RGB image with a white background (3 channels - R, G, B)
    height, width = mask1_padded.shape
    colored_mask = np.ones((height, width, 3))  # Initialize with all white (RGB: [1, 1, 1])

    # Set red for mask1
    colored_mask[mask1_boolean] = [1, 0, 0]  # Red color for mask1

    # Set green for mask2 where mask1 is not applied
    colored_mask[mask2_boolean & ~mask1_boolean] = [0, 1, 0]  # Green color for mask2

    # Set blue for mask3 where mask1 and mask2 are not applied
    colored_mask[mask3_boolean & ~mask1_boolean & ~mask2_boolean] = [0, 0, 1]  # Blue color for mask3

    # Show the combined mask with different colors on a white background
    #plt.figure(figsize=(10, 10))
    """
    plt.figure()
    plt.imshow(colored_mask, interpolation='none')
    plt.title("White Background with Yellow (Mask1), Red (Mask2), Blue (Mask3)")
    plt.axis('off')  # Turn off axis labels
    plt.show()
    """

    # Convert the colored mask (values in [0, 1]) to [0, 255] for saving
    colored_mask_uint8 = (colored_mask * 255).astype(np.uint8)
    # Convert the mask to an image and save it
    img = Image.fromarray(colored_mask_uint8)
    img.save('combined_colored_masks.png',dpi=(dpi, dpi))
    #-----

    masked_image1 = apply_mask_to_image(
        image_path=image_path_1,
        mask=mask1_padded,
        save_path='masked_image_1.jpg',
        target_dpi=dpi
    )

    masked_image2 = apply_mask_to_image(
        image_path=image_path_2,
        mask=mask2_padded,
        save_path='masked_image_2.jpg',
        target_dpi=dpi
    )

    masked_image3 = apply_mask_to_image(
        image_path=image_path_3,
        mask=mask3_padded,
        save_path='masked_image_3.jpg',
        target_dpi=dpi
    )


    # Combine images by taking the maximum value for each pixel
    combined_image_array_tmp = np.minimum(masked_image1, masked_image2)
    combined_image_array = np.minimum(combined_image_array_tmp, masked_image3)
    # Convert back to a PIL Image
    combined_image = Image.fromarray(combined_image_array.astype(np.uint8))

    #combined_image.save('combined_masked_image.jpg',dpi=(dpi, dpi))
    combined_image.save('combined_masked_image.png',dpi=(dpi, dpi))
    combined_image.show()

def moving_up_3(image_path_1, image_path_2, image_path_3, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius):
   
    mask1 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0, 0) #middle red
    mask2 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.55, 270) #up1 green
    mask3 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.1, 270) #up2 blue 
    
    """
    """
    _, mask1_padded, mask2_padded = combine_masks(mask1, mask2)
    combined_mask_tmp, mask1_padded, mask3_padded = combine_masks(mask1_padded, mask3)
    combined_mask, _, mask2_padded = combine_masks(combined_mask_tmp, mask2_padded)

    """
    plt.figure(figsize=(10, 10))
    plt.imshow(combined_mask, cmap='gray', interpolation='none')
    plt.title("Combined Circular Mask")
    plt.axis('off')
    plt.show()
    """

        #-----
    # Convert masks to boolean arrays (where 0 means inside the circle, 255 means background)
    mask1_boolean = mask1_padded == 0  # Black circles in mask1
    mask2_boolean = mask2_padded == 0  # Black circles in mask2
    mask3_boolean = mask3_padded == 0  # Black circles in mask3

    # Create an empty RGB image with a white background (3 channels - R, G, B)
    height, width = mask1_padded.shape
    colored_mask = np.ones((height, width, 3))  # Initialize with all white (RGB: [1, 1, 1])

    # Set red for mask1
    colored_mask[mask1_boolean] = [1, 0, 0]  # Red color for mask1

    # Set green for mask2 where mask1 is not applied
    colored_mask[mask2_boolean & ~mask1_boolean] = [0, 1, 0]  # Green color for mask2

    # Set blue for mask3 where mask1 and mask2 are not applied
    colored_mask[mask3_boolean & ~mask1_boolean & ~mask2_boolean] = [0, 0, 1]  # Blue color for mask3

    # Show the combined mask with different colors on a white background
    #plt.figure(figsize=(10, 10))
    """
    plt.figure()
    plt.imshow(colored_mask, interpolation='none')
    plt.title("White Background with Yellow (Mask1), Red (Mask2), Blue (Mask3)")
    plt.axis('off')  # Turn off axis labels
    plt.show()
    """

    # Convert the colored mask (values in [0, 1]) to [0, 255] for saving
    colored_mask_uint8 = (colored_mask * 255).astype(np.uint8)
    # Convert the mask to an image and save it
    img = Image.fromarray(colored_mask_uint8)
    img.save('combined_colored_masks.png',dpi=(dpi, dpi))
    #-----

    masked_image1 = apply_mask_to_image(
        image_path=image_path_1,
        mask=mask1_padded,
        save_path='masked_image_1.jpg',
        target_dpi=dpi
    )

    masked_image2 = apply_mask_to_image(
        image_path=image_path_2,
        mask=mask2_padded,
        save_path='masked_image_2.jpg',
        target_dpi=dpi
    )

    masked_image3 = apply_mask_to_image(
        image_path=image_path_3,
        mask=mask3_padded,
        save_path='masked_image_3.jpg',
        target_dpi=dpi
    )


    # Combine images by taking the maximum value for each pixel
    combined_image_array_tmp = np.minimum(masked_image1, masked_image2)
    combined_image_array = np.minimum(combined_image_array_tmp, masked_image3)
    # Convert back to a PIL Image
    combined_image = Image.fromarray(combined_image_array.astype(np.uint8))

    #combined_image.save('combined_masked_image.jpg',dpi=(dpi, dpi))
    combined_image.save('combined_masked_image.png',dpi=(dpi, dpi))
    combined_image.show()

#square experiments
def four_images_square(image_path_1, image_path_2, image_path_3,image_path_4, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius):

    #mask1 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0, 0)
    #mask2 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 180)
    #mask3 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 270)
    #mask4 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, -1.5, 270)
    """
    mask1 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.5, 0) #right
    mask2 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 180) #left
    mask3 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 270) #up
    mask4 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.5, 90) #down
    """
    mask1 = create_square_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius, 1.5, 0) #right red
    mask2 = create_square_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius, 0.5, 180) #left green
    mask3 = create_square_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius, 0.5, 270) #up blue
    mask4 = create_square_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius, 1.5, 90) #down yellow



    combined_mask_tmp1, mask1_padded, mask4_padded = combine_masks(mask1, mask4)
    combined_mask_tmp2, _, mask2_padded = combine_masks(combined_mask_tmp1, mask2)
    combined_mask, _, mask3_padded = combine_masks(combined_mask_tmp2, mask3)
    _, _, mask1_padded = combine_masks(combined_mask, mask1)
    _, _, mask2_padded = combine_masks(combined_mask, mask2)
    _, _, mask4_padded = combine_masks(combined_mask, mask4)


    #plt.figure(figsize=(10, 10))
    """
    plt.figure()
    plt.imshow(combined_mask, cmap='gray', interpolation='none')
    plt.title("Combined Circular Mask")
    plt.axis('off')
    plt.show()
    """
    #black_mask_uint8 = (combined_mask * 255).astype(np.uint8)
    #img_black = Image.fromarray(black_mask_uint8)
    #img_black.save('combined_mask_black.png',dpi=(dpi, dpi))
    #-----
    # Convert masks to boolean arrays (where 0 means inside the circle, 255 means background)
    mask1_boolean = mask1_padded == 0  # Black circles in mask1
    mask2_boolean = mask2_padded == 0  # Black circles in mask2
    mask3_boolean = mask3_padded == 0  # Black circles in mask3
    mask4_boolean = mask4_padded == 0  # Black circles in mask4

    # Create an empty RGB image with a white background (3 channels - R, G, B)
    height, width = mask1_padded.shape
    colored_mask = np.ones((height, width, 3))  # Initialize with all white (RGB: [1, 1, 1])

    # Set red for mask1
    colored_mask[mask1_boolean] = [1, 0, 0]  # Red color for mask1

    # Set green for mask2 where mask1 is not applied
    colored_mask[mask2_boolean & ~mask1_boolean] = [0, 1, 0]  # Green color for mask2

    # Set blue for mask3 where mask1 and mask2 are not applied
    colored_mask[mask3_boolean & ~mask1_boolean & ~mask2_boolean] = [0, 0, 1]  # Blue color for mask3

    # Set yellow for mask4 where mask1, mask2, and mask3 are not applied
    colored_mask[mask4_boolean & ~mask1_boolean & ~mask2_boolean & ~mask3_boolean] = [1, 1, 0]  # Yellow color for mask4

    # Show the combined mask with different colors on a white background
    #plt.figure(figsize=(10, 10))
    """
    plt.figure()
    plt.imshow(colored_mask, interpolation='none')
    plt.title("White Background with Red (Mask1), Green (Mask2), Blue (Mask3), and Yellow (Mask4)")
    plt.axis('off')  # Turn off axis labels
    plt.show()
    """

    # Convert the colored mask (values in [0, 1]) to [0, 255] for saving
    colored_mask_uint8 = (colored_mask * 255).astype(np.uint8)
    # Convert the mask to an image and save it
    img = Image.fromarray(colored_mask_uint8)
    img.save('combined_colored_masks.png',dpi=(dpi, dpi))
    #-----

    masked_image1 = apply_mask_to_image(
        image_path=image_path_1,
        mask=mask1_padded,
        save_path='masked_image_1.jpg',
        target_dpi=dpi
    )

    masked_image2 = apply_mask_to_image(
        image_path=image_path_2,
        mask=mask2_padded,
        save_path='masked_image_2.jpg',
        target_dpi=dpi
    )

    masked_image3 = apply_mask_to_image(
        image_path=image_path_3,
        mask=mask3_padded,
        save_path='masked_image_3.jpg',
        target_dpi=dpi
    )

    masked_image4 = apply_mask_to_image(
        image_path=image_path_4,
        mask=mask4_padded,
        save_path='masked_image_4.jpg',
        target_dpi=dpi
    )


    # Combine images by taking the maximum value for each pixel
    combined_image_array_tmp1 = np.minimum(masked_image1, masked_image2)
    combined_image_array_tmp2 = np.minimum(combined_image_array_tmp1, masked_image3)
    combined_image_array = np.minimum(combined_image_array_tmp2, masked_image4)
    # Convert back to a PIL Image
    combined_image = Image.fromarray(combined_image_array.astype(np.uint8))

    #combined_image.save('combined_masked_image.jpg',dpi=(dpi, dpi))
    combined_image.save('combined_masked_image.png',dpi=(dpi, dpi))
    combined_image.show()

def five_images_square(image_path_1, image_path_2, image_path_3,image_path_4, image_path_5, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius):

    """
    1 Center Circle:
    move_dist_mm = 0, angle_degrees = 0 (No displacement)

    2 Right Circle:
    move_dist_mm = some_positive_value, angle_degrees = 0 (Move right)

    3 Left Circle:
    move_dist_mm = some_positive_value, angle_degrees = 180 (Move left)

    4 Up Circle:
    move_dist_mm = some_positive_value, angle_degrees = 270 (Move up)

    5 Down Circle:
    move_dist_mm = some_positive_value, angle_degrees = 90 (Move down)
    """

    
    mask1 = create_square_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius, 0, 0) #center
    mask2 = create_square_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius, 1.4, 0) #1.3 right
    mask3 = create_square_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius, 0.5, 180) #0.5 left
    mask4 = create_square_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius, 0.5, 270) #up
    mask5 = create_square_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius, 1.5, 90) #down
    
    #mask1 = create_circle_on_elliptical_base_mm(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0, 0)
    #mask2 = create_circle_on_elliptical_base_mm(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.5, 0)
    #mask3 = create_circle_on_elliptical_base_mm(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 180)
    #mask4 = create_circle_on_elliptical_base_mm(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.5, 270)
    #mask5 = create_circle_on_elliptical_base_mm(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.5, 90)



    combined_mask_tmp1, mask1_padded, mask4_padded = combine_masks(mask1, mask4)
    combined_mask_tmp2, _, mask2_padded = combine_masks(combined_mask_tmp1, mask2)
    combined_mask_tmp3, _, mask5_padded = combine_masks(combined_mask_tmp2, mask5)
    combined_mask, _, mask3_padded = combine_masks(combined_mask_tmp3, mask3)
    _, _, mask1_padded = combine_masks(combined_mask, mask1)
    _, _, mask2_padded = combine_masks(combined_mask, mask2)
    _, _, mask4_padded = combine_masks(combined_mask, mask4)
    _, _, mask5_padded = combine_masks(combined_mask, mask5)

    """
    #plt.figure(figsize=(10, 10))
    plt.imshow(combined_mask, cmap='gray', interpolation='none')
    plt.title("Combined Circular Mask")
    plt.axis('off')
    plt.show()
    """
    #-----
    # Convert masks to boolean arrays (where 0 means inside the circle, 255 means background)
    mask1_boolean = mask1_padded == 0  # Black circles in mask1
    mask2_boolean = mask2_padded == 0  # Black circles in mask2
    mask3_boolean = mask3_padded == 0  # Black circles in mask3
    mask4_boolean = mask4_padded == 0  # Black circles in mask4
    mask5_boolean = mask5_padded == 0  # Black circles in mask5

    # Create an empty RGB image with a white background (3 channels - R, G, B)
    height, width = mask1_padded.shape
    colored_mask = np.ones((height, width, 3))  # Initialize with all white (RGB: [1, 1, 1])

    # Set red for mask1
    colored_mask[mask1_boolean] = [1, 0, 0]  # Red color for mask1

    # Set green for mask2 where mask1 is not applied
    colored_mask[mask2_boolean & ~mask1_boolean] = [0, 1, 0]  # Green color for mask2

    # Set blue for mask3 where mask1 and mask2 are not applied
    colored_mask[mask3_boolean & ~mask1_boolean & ~mask2_boolean] = [0, 0, 1]  # Blue color for mask3

    # Set yellow for mask4 where mask1, mask2, and mask3 are not applied
    colored_mask[mask4_boolean & ~mask1_boolean & ~mask2_boolean & ~mask3_boolean] = [1, 1, 0]  # Yellow color for mask4

    # Set magenta for mask5 where masks 1, 2, 3, and 4 are not applied
    colored_mask[mask5_boolean & ~mask1_boolean & ~mask2_boolean & ~mask3_boolean & ~mask4_boolean] = [1, 0, 1]  # Magenta color for mask5

    # Show the combined mask with different colors on a white background
    #plt.figure(figsize=(10, 10))
    """
    plt.imshow(colored_mask, interpolation='none')
    plt.title("White Background with Red (Mask1), Green (Mask2), Blue (Mask3), Yellow (Mask4) amd Magenta (Mask5)")
    plt.axis('off')  # Turn off axis labels
    plt.show()
    """

    # Convert the colored mask (values in [0, 1]) to [0, 255] for saving
    colored_mask_uint8 = (colored_mask * 255).astype(np.uint8)
    # Convert the mask to an image and save it
    img = Image.fromarray(colored_mask_uint8)
    img.save('combined_colored_masks.png',dpi=(dpi, dpi))
    #-----

    masked_image1 = apply_mask_to_image(
        image_path=image_path_1,
        mask=mask1_padded,
        save_path='masked_image_1.jpg',
        target_dpi=dpi
    )

    masked_image2 = apply_mask_to_image(
        image_path=image_path_2,
        mask=mask2_padded,
        save_path='masked_image_2.jpg',
        target_dpi=dpi
    )

    masked_image3 = apply_mask_to_image(
        image_path=image_path_3,
        mask=mask3_padded,
        save_path='masked_image_3.jpg',
        target_dpi=dpi
    )

    masked_image4 = apply_mask_to_image(
        image_path=image_path_4,
        mask=mask4_padded,
        save_path='masked_image_4.jpg',
        target_dpi=dpi
    )

    masked_image5 = apply_mask_to_image(
        image_path=image_path_5,
        mask=mask5_padded,
        save_path='masked_image_5.jpg',
        target_dpi=dpi
    )


    # Combine images by taking the maximum value for each pixel
    combined_image_array_tmp1 = np.minimum(masked_image1, masked_image2)
    combined_image_array_tmp2 = np.minimum(combined_image_array_tmp1, masked_image3)
    combined_image_array_tmp3 = np.minimum(combined_image_array_tmp2, masked_image4)
    combined_image_array = np.minimum(combined_image_array_tmp3, masked_image5)
    # Convert back to a PIL Image
    combined_image = Image.fromarray(combined_image_array.astype(np.uint8))

    #combined_image.save('combined_masked_image.jpg',dpi=(dpi, dpi))
    combined_image.save('combined_masked_image.png',dpi=(dpi, dpi))
    combined_image.show()

def moving_right_square(image_path_1, image_path_2, image_path_3,image_path_4, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius):

    #mask1 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0, 0) #middle
    #mask2 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0.92, 0) #right1
    #mask3 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.84, 0) #right2
    #mask4 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 2.76, 0) #right3
   
    mask1 = create_square_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius, 0, 0) #middle
    mask2 = create_square_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius, 1.84, 0) #right1
    mask3 = create_square_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius, 3.68, 0) #right2
    mask4 = create_square_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius, 5.52, 0) #right3
    """
    mask1 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 0, 0) #middle
    mask2 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 1.8, 0) #right1
    mask3 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 3.6, 0) #right2
    mask4 = create_circle_on_elliptical_base(dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius, 5.4, 0) #right3
    """


    combined_mask_tmp1, mask1_padded, mask4_padded = combine_masks(mask1, mask4)
    combined_mask_tmp2, _, mask2_padded = combine_masks(combined_mask_tmp1, mask2)
    combined_mask, _, mask3_padded = combine_masks(combined_mask_tmp2, mask3)
    _, _, mask1_padded = combine_masks(combined_mask, mask1)
    _, _, mask2_padded = combine_masks(combined_mask, mask2)
    _, _, mask4_padded = combine_masks(combined_mask, mask4)


    #plt.figure(figsize=(10, 10))
    """
    plt.imshow(combined_mask, cmap='gray', interpolation='none')
    plt.title("Combined Circular Mask")
    plt.axis('off')
    plt.show()
    """
    #-----
    # Convert masks to boolean arrays (where 0 means inside the circle, 255 means background)
    mask1_boolean = mask1_padded == 0  # Black circles in mask1
    mask2_boolean = mask2_padded == 0  # Black circles in mask2
    mask3_boolean = mask3_padded == 0  # Black circles in mask3
    mask4_boolean = mask4_padded == 0  # Black circles in mask4

    # Create an empty RGB image with a white background (3 channels - R, G, B)
    height, width = mask1_padded.shape
    colored_mask = np.ones((height, width, 3))  # Initialize with all white (RGB: [1, 1, 1])

    # Set red for mask1
    colored_mask[mask1_boolean] = [1, 0, 0]  # Red color for mask1

    # Set green for mask2 where mask1 is not applied
    colored_mask[mask2_boolean & ~mask1_boolean] = [0, 1, 0]  # Green color for mask2

    # Set blue for mask3 where mask1 and mask2 are not applied
    colored_mask[mask3_boolean & ~mask1_boolean & ~mask2_boolean] = [0, 0, 1]  # Blue color for mask3

    # Set yellow for mask4 where mask1, mask2, and mask3 are not applied
    colored_mask[mask4_boolean & ~mask1_boolean & ~mask2_boolean & ~mask3_boolean] = [1, 1, 0]  # Yellow color for mask4

    # Show the combined mask with different colors on a white background
    #plt.figure(figsize=(10, 10))
    """
    plt.imshow(colored_mask, interpolation='none')
    plt.title("White Background with Red (Mask1), Green (Mask2), Blue (Mask3), and Yellow (Mask4)")
    plt.axis('off')  # Turn off axis labels
    plt.show()
    """

    # Convert the colored mask (values in [0, 1]) to [0, 255] for saving
    colored_mask_uint8 = (colored_mask * 255).astype(np.uint8)
    # Convert the mask to an image and save it
    img = Image.fromarray(colored_mask_uint8)
    img.save('combined_colored_masks.png',dpi=(dpi, dpi))
    #-----

    masked_image1 = apply_mask_to_image(
        image_path=image_path_1,
        mask=mask1_padded,
        save_path='masked_image_1.jpg',
        target_dpi=dpi
    )

    masked_image2 = apply_mask_to_image(
        image_path=image_path_2,
        mask=mask2_padded,
        save_path='masked_image_2.jpg',
        target_dpi=dpi
    )

    masked_image3 = apply_mask_to_image(
        image_path=image_path_3,
        mask=mask3_padded,
        save_path='masked_image_3.jpg',
        target_dpi=dpi
    )

    masked_image4 = apply_mask_to_image(
        image_path=image_path_4,
        mask=mask4_padded,
        save_path='masked_image_4.jpg',
        target_dpi=dpi
    )


    # Combine images by taking the maximum value for each pixel
    combined_image_array_tmp1 = np.minimum(masked_image1, masked_image2)
    combined_image_array_tmp2 = np.minimum(combined_image_array_tmp1, masked_image3)
    combined_image_array = np.minimum(combined_image_array_tmp2, masked_image4)
    # Convert back to a PIL Image
    combined_image = Image.fromarray(combined_image_array.astype(np.uint8))

    #combined_image.save('combined_masked_image.jpg',dpi=(dpi, dpi))
    combined_image.save('combined_masked_image.png',dpi=(dpi, dpi))
    combined_image.show()

def main(size_mm,is_type_text,is_circle,type, image_paths):
    # Constants: the values of lens
    dpi=300
    shift_odd_rows=1.84
    dist_between_circles_in_row=3.68
    vertical_dist=1.225
    ellipse_radius_width=1.2  # Width of the ellipse
    ellipse_radius_length=0.9  # Length of the ellipse
    circle_radius=0.35  # Radius of the circle - what we always use
    #square_side = 0.6
    square_radius = 0.3 #0.3

    if is_type_text:    
        type_text()
        num_images = count_files_in_directory(f"images/typed_text") 
        dir = 'typed_text'
        
        # inputs images:
        if num_images == 1:
            #1 typed text
            image_path_1='images/' + dir +'/word_1.png'  
            one_image(image_path_1, dpi,size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius)

        #2 typed text
        elif num_images == 2:
            image_path_1='images/' + dir +'/word_1.png' #Center 
            image_path_2='images/' +dir+'/word_2.png' #right 1
            two_images(image_path_1, image_path_2, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius)

        #3 typed text
        elif num_images == 3:
            image_path_1='images/' + dir +'/word_1.png' #Center 
            image_path_2='images/' +dir+'/word_2.png' #right1
            image_path_3='images/' +dir+'/word_3.png' #right2
            moving_right_3(image_path_1, image_path_2, image_path_3, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius)
        
        #4 typed text
        elif num_images == 4:
            image_path_1='images/' + dir +'/word_1.png' #center
            image_path_2='images/' +dir+'/word_2.png' #right1
            image_path_3='images/' +dir+'/word_3.png' #right2
            image_path_4='images/' +dir+'/word_4.png' #right3
            moving_right_4(image_path_1, image_path_2, image_path_3,image_path_4, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius)

    else:
        # inputs images:
        image_path_1=image_paths[0]
        image_path_2=image_paths[1]
        image_path_3=image_paths[2]
        image_path_4=image_paths[3]
        image_path_5=image_paths[4]

        #one_image(image_path, dpi,size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius)
        #two_images(image_path_1, image_path_2, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius)
        #three_images(image_path_1, image_path_2, image_path_3, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius)
        
        #circle
        if is_circle:
            if type == '3':
                three_images(image_path_1, image_path_2, image_path_3, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius)
            elif type == '4':
                four_images(image_path_1, image_path_2, image_path_3,image_path_4, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius)
            elif type == '5':
                five_images(image_path_1, image_path_2, image_path_3,image_path_4, image_path_5, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius)
            elif type == 'moving_right_4':
                moving_right_4(image_path_1, image_path_2, image_path_3,image_path_4, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius)
            elif type == 'moving_right_3':
                moving_right_3(image_path_1, image_path_2, image_path_3, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius)
            elif type == 'moving_up_3':
                moving_up_3(image_path_1, image_path_2, image_path_3, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, circle_radius)

        #square
        else: #square
            if type == '4':
                four_images_square(image_path_1, image_path_2, image_path_3,image_path_4, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius)
            if type == '5':    
                five_images_square(image_path_1, image_path_2, image_path_3,image_path_4, image_path_5, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius)
            if type == 'moving_right':
                moving_right_square(image_path_1, image_path_2, image_path_3,image_path_4, dpi, size_mm, shift_odd_rows, dist_between_circles_in_row, vertical_dist, square_radius)

#--------------------------------------------------
# main - change parameters:
if __name__ == '__main__':

    #Inputs: change the parameters below to your own settings
    
    #size of lense
    size_mm = 100 #120

    # is_type_text is True if you want the interactive program, insert text.
    # False to insert images  
    is_type_text = True
    # is_circle - True to choose mask of circles, False of squares
    is_circle = True
    #type - choose type of task: 
    # 'moving_right_4' - sliding image, shift to right 4 times (4 images) 
    # 'moving_right_3'- sliding image, shift to right 3 times (3 images) 
    # 'moving_up_3'- sliding image, go up 3 times (3 images) 
    # '3' - 
    # '4' - right, left, up, down
    # '5' - center, right, left, up, down
    type = '4' 

    #Choose images from the comments bellow:
    #4 Felix
    image_path_1='images/Felix/felix_right.jpg' #Right Circle
    image_path_2='images/Felix/felix_left.jpg' #Left Circle
    image_path_3='images/Felix/felix_up.jpg' #Up Circle
    image_path_4='images/Felix/felix_down.jpg' #Down Circle
    image_path_5=None

    #----Do not change from here----

    image_paths = []
    image_paths.append(image_path_1)
    image_paths.append(image_path_2)
    image_paths.append(image_path_3)
    image_paths.append(image_path_4)
    image_paths.append(image_path_5)

    main(size_mm,is_type_text,is_circle,type, image_paths)
    
    """
    images options:

    #4 directions
    image_path_1='images/Directions/right.jpg' #Right Circle
    image_path_2='images/Directions/left.jpg' #Left Circle
    image_path_3='images/Directions/up.jpg' #Up Circle
    image_path_4='images/Directions/down.jpg' #Down Circle
    image_path_5=None

    #4 Felix
    image_path_1='images/Felix/felix_right.jpg' #Right Circle
    image_path_2='images/Felix/felix_left.jpg' #Left Circle
    image_path_3='images/Felix/felix_up.jpg' #Up Circle
    image_path_4='images/Felix/felix_down.jpg' #Down Circle
    image_path_5=None

    #4 moving right Monty
    image_path_1='images/Monty/monty_1.jpg' #center 
    image_path_2='images/Monty/monty_2.jpg' #right1 
    image_path_3='images/Monty/monty_3.jpg' #right2 
    image_path_4='images/Monty/monty_4.jpg' #right3
    image_path_5=None

    #3 moving up basketball
    image_path_1='images/basketball/ball_1.jpg' #center 
    image_path_2='images/basketball/ball_3.jpg' #right1 
    image_path_3='images/basketball/ball_4.jpg' #right2 
    image_path_4=None
    image_path_5=None

    #4 sliding right matter
    image_path_1='images/Matter_of_pers/1.jpg' #Left 
    image_path_2='images/Matter_of_pers/2.jpg' #Center 
    image_path_3='images/Matter_of_pers/3.jpg' #Right
    image_path_4='images/Matter_of_pers/4.jpg' #Right
    image_path_5=None

    #4 colors
    image_path_1='images/colors/green.png' #Right Circle
    image_path_2='images/colors/darkblue.png' #Left Circle
    image_path_3='images/colors/red.png' #Up Circle
    image_path_4='images/colors/yellow.png' #Down Circle
    image_path_5=None

    #5 colors
    image_path_1='images/colors/darkblue.png' #Center Circle
    image_path_2= 'images/colors/lightblue.png' #Right Circle
    image_path_3='images/colors/red.png' #Left Circle
    image_path_4='images/colors/yellow.png' #Up Circle
    image_path_5='images/colors/green.png' #Down Circle
    """ 

