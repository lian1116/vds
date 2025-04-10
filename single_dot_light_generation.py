from model import *

# Example usage
if __name__ == "__main__":
    # Input image path
    image_path = "darkened_image.jpg"
    image = cv2.imread(image_path)
    # Get image dimensions
    height, width = image.shape[:2]
    # Create a pure black mask image
    mask = np.zeros_like(image)
    # Randomly generate trapezoid parameters
    min_side = min(height, width)
    # Set parameters
    print(min_side)
    dot_intensity = 5  # Glow intensity multiplier
    glow_size_range = (min_side // 1.5, min_side)  # Glow size range
    center_brightness = 255  # Brightness of the center of the glow (0-255)
    # r, g, b = 217, 58, 54  # Glow color (yellow)
    # r, g, b = 76, 179, 57  # Glow color (yellow)
    # r, g, b = 36, 75, 152  # Glow color (yellow)

    r = g = b = 80

    dot_intensity = 2.3  # Glow intensity multiplier
    # dot_intensity = 2.5  # Glow intensity multiplier
    # dot_intensity = 2.7  # Glow intensity multiplier
    # dot_intensity = 3  # Glow intensity multiplier
    # dot_intensity = 3.5  # Glow intensity multiplier

    input_dir = "0"
    for i, filename in enumerate(os.listdir(input_dir)):
        # Construct the full input file path
        input_path = os.path.join(input_dir, filename)
        print(input_path)
        generate_0(input_path, 20, glow_size_range, center_brightness)
    #
    # output_image = add_oval_laser(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, False)
    # if output_image is not None:
    #     # Display the result
    #     cv2.imshow("Random Laser Dot with Glow", output_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    #     # Save the result
    #     cv2.imwrite("output_random_laser_dot.jpg", output_image)
    #
