import cv2
import numpy as np
import random
import os

def add_single_laser_dot(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b):
    """
    Generate a single dot laser pattern in the center of the image and add a glow effect.
    :param image_path: Path to the input image
    :param dot_radius_range: Range of dot radius (min, max)
    :param dot_intensity: Intensity multiplier for the dot
    :param glow_size_range: Range for the glow size (min, max)
    :param center_brightness: Brightness at the center of the spot (0-255)
    :param r: Red channel value of the dot (0-255)
    :param g: Green channel value of the dot (0-255)
    :param b: Blue channel value of the dot (0-255)
    :return: Image with laser pattern added
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image, please check the path.")
        return None

    # Get image dimensions
    height, width = image.shape[:2]
    min_xy = min(image.shape[:2][0], image.shape[:2][1])

    # Create a black mask image
    mask = np.zeros_like(image)

    # Randomly generate dot radius
    dot_radius = random.randint(min_xy // 4, min_xy // 2)

    dot_place_x = random.randint(image.shape[:2][0] // 5, image.shape[:2][1] // 5)
    dot_place_y = random.randint(image.shape[:2][0] // 5, image.shape[:2][1] // 5)

    # Randomly generate the center of the dot
    center_x = random.randint(dot_place_x, width - dot_place_x)
    center_y = random.randint(dot_place_y, height - dot_place_y)

    # Draw the dot at the random position
    center_color = (int(b * center_brightness / 255), int(g * center_brightness / 255), int(r * center_brightness / 255))
    cv2.circle(mask, (center_x, center_y), dot_radius, center_color, -1)  # Draw the dot

    # Randomly generate the glow size (must be a positive odd number)
    glow_size = random.randint(glow_size_range[0], glow_size_range[1])
    if glow_size % 2 == 0:  # Ensure it's odd
        glow_size += 1

    # Apply Gaussian blur to the mask to simulate the glow effect
    mask_blur = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)

    # Overlay the glow effect onto the original image
    result = cv2.addWeighted(image, 1.0, mask_blur, dot_intensity, 0)

    return result

def add_trapezoid_laser(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, use_gaussian_blur):
    """
    Generate a trapezoidal laser pattern in the center of the image and add a glow effect.
    :param image_path: Path to the input image
    :param dot_intensity: Intensity multiplier for the dot
    :param glow_size_range: Range for the glow size (min, max)
    :param center_brightness: Brightness at the center of the spot (0-255)
    :param r: Red channel value of the dot (0-255)
    :param g: Green channel value of the dot (0-255)
    :param b: Blue channel value of the dot (0-255)
    :return: Image with laser pattern added
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image, please check the path.")
        return None

    # Get image dimensions
    height, width = image.shape[:2]

    # Create a black mask image
    mask = np.zeros_like(image)

    # Randomly generate trapezoid parameters
    min_side = min(height, width)
    trapezoid_width = random.randint(min_side // 3, min_side // 1.2)  # Trapezoid base width
    trapezoid_height = random.randint(min_side // 3, min_side // 1.2)  # Trapezoid height
    trapezoid_shift = random.randint(-trapezoid_width // 4, trapezoid_width // 4)  # Trapezoid tilt

    # Randomly generate trapezoid position
    center_x = random.randint(trapezoid_width // 2, width - trapezoid_width // 2)
    center_y = random.randint(trapezoid_height // 2, height - trapezoid_height // 2)

    # Calculate the four vertices of the trapezoid
    top_left = (center_x - trapezoid_width // 2 + trapezoid_shift, center_y - trapezoid_height // 2)
    top_right = (center_x + trapezoid_width // 2 + trapezoid_shift, center_y - trapezoid_height // 2)
    bottom_right = (center_x + trapezoid_width // 2, center_y + trapezoid_height // 2)
    bottom_left = (center_x - trapezoid_width // 2, center_y + trapezoid_height // 2)

    # Convert the vertices to a NumPy array
    trapezoid_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

    # Draw the trapezoid on the mask
    center_color = (int(b * center_brightness / 255), int(g * center_brightness / 255), int(r * center_brightness / 255))
    cv2.fillPoly(mask, [trapezoid_points], center_color)  # Draw the trapezoid

    if use_gaussian_blur:
        # Randomly generate the glow size (must be a positive odd number)
        glow_size = random.randint(glow_size_range[0], glow_size_range[1])
        if glow_size % 2 == 0:  # Ensure it's odd
            glow_size += 1

        # Apply Gaussian blur to the mask to simulate the glow effect
        mask_blur = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)
    else:
        # Do not use Gaussian blur, use the original mask
        mask_blur = mask

    # Overlay the glow effect onto the original image
    result = cv2.addWeighted(image, 1.0, mask_blur, dot_intensity, 0)

    return result

def add_oval_laser(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, use_gaussian_blur=True):
    """
    Generate an oval laser pattern in the center of the image and add a glow effect.
    :param image_path: Path to the input image
    :param dot_intensity: Intensity multiplier for the dot
    :param glow_size_range: Range for the glow size (min, max)
    :param center_brightness: Brightness at the center of the spot (0-255)
    :param r: Red channel value of the dot (0-255)
    :param g: Green channel value of the dot (0-255)
    :param b: Blue channel value of the dot (0-255)
    :param use_gaussian_blur: Whether to use Gaussian blur (default is True)
    :return: Image with laser pattern added
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image, please check the path.")
        return None

    # Get image dimensions
    height, width = image.shape[:2]

    # Create a black mask image
    mask = np.zeros_like(image)

    # Randomly generate oval parameters
    min_side = min(height, width)
    ellipse_width = random.randint(min_side // 3, min_side // 1.2)  # Oval width
    ellipse_height = random.randint(min_side // 3, min_side // 1.2)  # Oval height
    angle = random.randint(0, 180)  # Oval rotation angle

    # Randomly generate oval position
    center_x = random.randint(ellipse_width // 2, width - ellipse_width // 2)
    center_y = random.randint(ellipse_height // 2, height - ellipse_height // 2)

    # Draw the oval at the random position
    center_color = (int(b * center_brightness / 255), int(g * center_brightness / 255), int(r * center_brightness / 255))
    cv2.ellipse(
        mask,
        (center_x, center_y),  # Oval center
        (ellipse_width // 2, ellipse_height // 2),  # Long and short axes
        angle,  # Rotation angle
        0,  # Start angle
        360,  # End angle
        center_color,  # Color
        -1  # Fill the oval
    )

    # Apply Gaussian blur if necessary
    if use_gaussian_blur:
        # Randomly generate the glow size (must be a positive odd number)
        glow_size = random.randint(glow_size_range[0], glow_size_range[1])
        if glow_size % 2 == 0:  # Ensure it's odd
            glow_size += 1

        # Apply Gaussian blur to the mask to simulate the glow effect
        mask_blur = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)
    else:
        # Do not use Gaussian blur, use the original mask
        mask_blur = mask

    # Overlay the glow effect onto the original image
    result = cv2.addWeighted(image, 1.0, mask_blur, dot_intensity, 0)

    return result

def generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light, flag):
    i = 0
    while i < num // 2:
        output_image = add_trapezoid_laser(
            image_path,
            dot_intensity=dot_intensity,
            glow_size_range=glow_size_range,
            center_brightness=center_brightness,
            r=r,
            g=g,
            b=b,
            use_gaussian_blur=flag
        )

        if output_image is not None:
            parts = image_path.split("_")  # Split by "_"
            number = parts[1].split(".")[0]  # Extract "0" and remove ".jpg"
            cv2.imwrite("jiguang\\{}\\{}_{}_{}.jpg".format(type_light, number, type_light, i), output_image)
            i += 1
        else:
            print("Error: Failed to generate image.")
            break
    while i < num:
        output_image = add_oval_laser(
            image_path,
            dot_intensity=dot_intensity,
            glow_size_range=glow_size_range,
            center_brightness=center_brightness,
            r=r,
            g=g,
            b=b,
            use_gaussian_blur=flag
        )

        if output_image is not None:
            parts = image_path.split("_")  # Split by "_"
            number = parts[1].split(".")[0]  # Extract "0" and remove ".jpg"
            cv2.imwrite("jiguang\\{}\\{}_{}_{}.jpg".format(type_light, number, type_light, i), output_image)
            i += 1
        else:
            print("Error: Failed to generate image.")
            break

def generate_0(image_path, num, glow_size_range, center_brightness):
    type_light = 1
    while type_light <= 8:
        if type_light == 1:
            r, g, b = 217, 58, 54  # Dot color (yellow)
            dot_intensity = 5
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light, flag=True)
        elif type_light == 2:
            r, g, b = 76, 179, 57  # Dot color (yellow)
            dot_intensity = 5
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light, flag=True)
        elif type_light == 3:
            r, g, b = 36, 75, 152  # Dot color (yellow)
            dot_intensity = 5
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light, flag=True)
        elif type_light == 4:
            r = g = b = 80
            dot_intensity = 2.3  # Glow intensity multiplier
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light, flag=False)
        elif type_light == 5:
            r = g = b = 80
            dot_intensity = 2.5  # Glow intensity multiplier
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light, flag=False)
        elif type_light == 6:
            r = g = b = 80
            dot_intensity = 2.7  # Glow intensity multiplier
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light, flag=False)
        elif type_light == 7:
            r = g = b = 80
            dot_intensity = 3  # Glow intensity multiplier
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light, flag=False)
        elif type_light == 8:
            r = g = b = 80
            dot_intensity = 3.5  # Glow intensity multiplier
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light, flag=False)

        type_light += 1
