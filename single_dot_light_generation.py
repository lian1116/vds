from model import *

# 使用示例
if __name__ == "__main__":
    # 输入图片路径
    image_path = "darkened_image.jpg"
    image = cv2.imread(image_path)
    # 获取图片尺寸
    height, width = image.shape[:2]
    # 创建一个纯黑色的掩码图像
    mask = np.zeros_like(image)
    # 随机生成梯形的参数
    min_side = min(height, width)
    # 设置参数
    print(min_side)
    dot_intensity = 5  # 光晕的亮度增强倍数
    glow_size_range = (min_side//1.5, min_side)  # 光晕大小范围
    center_brightness = 255  # 光斑中心的亮度（0-255）
    # r, g, b = 217, 58, 54  # 光点的颜色（黄色）
    # r, g, b = 76, 179, 57  # 光点的颜色（黄色）
    # r, g, b = 36, 75, 152  # 光点的颜色（黄色）

    r= g = b = 80

    dot_intensity = 2.3  # 光晕的亮度增强倍数
    # dot_intensity = 2.5 # 光晕的亮度增强倍数
    # dot_intensity = 2.7  # 光晕的亮度增强倍数
    # dot_intensity = 3  # 光晕的亮度增强倍数
    # dot_intensity = 3.5  # 光晕的亮度增强倍数

    input_dir = "0"
    for i, filename in enumerate(os.listdir(input_dir)):
        # 构建输入文件的完整路径
        input_path = os.path.join(input_dir, filename)
        print(input_path)
        generate_0(input_path, 20, glow_size_range,center_brightness)
    #
    # output_image = add_oval_laser(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b,False)
    # if output_image is not None:
    #     # 显示结果
    #     cv2.imshow("Random Laser Dot with Glow", output_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    #     # 保存结果
    #     cv2.imwrite("output_random_laser_dot.jpg", output_image)
    #
