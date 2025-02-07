import cv2
import numpy as np
import random
import os
def add_single_laser_dot(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b):
    """
    在图片的中间生成单个点状激光图样，并添加光晕效果
    :param image_path: 输入图片的路径
    :param dot_radius_range: 光点半径的范围（最小值，最大值）
    :param dot_intensity: 光点的亮度增强倍数
    :param glow_size_range: 光晕大小的范围（最小值，最大值）
    :param center_brightness: 光斑中心的亮度（0-255）
    :param r: 光点的红色通道值（0-255）
    :param g: 光点的绿色通道值（0-255）
    :param b: 光点的蓝色通道值（0-255）
    :return: 添加激光图样后的图片
    """
    # 加载图片
    image = cv2.imread(image_path)
    if image is None:
        print("错误：无法加载图片，请检查路径是否正确。")
        return None

    # 获取图片尺寸
    height, width = image.shape[:2]
    min_xy = min(image.shape[:2][0],image.shape[:2][1])
    # 创建一个纯黑色的掩码图像
    mask = np.zeros_like(image)

    # 随机生成光斑半径
    dot_radius = random.randint(min_xy//4,min_xy//2)

    dot_place_x = random.randint(image.shape[:2][0]//5,image.shape[:2][1]//5)
    dot_place_y = random.randint(image.shape[:2][0]//5,image.shape[:2][1]//5)
    # 随机生成光斑中心位置
    center_x = random.randint(dot_place_x, width - dot_place_x)
    center_y = random.randint(dot_place_y, height - dot_place_y)

    # 在随机位置绘制光点
    center_color = (int(b * center_brightness / 255), int(g * center_brightness / 255), int(r * center_brightness / 255))
    cv2.circle(mask, (center_x, center_y), dot_radius, center_color, -1)  # 绘制光点

    # 随机生成光晕大小（必须是正奇数）
    glow_size = random.randint(glow_size_range[0], glow_size_range[1])
    if glow_size % 2 == 0:  # 确保是奇数
        glow_size += 1

    # 对掩码进行高斯模糊，模拟光晕效果
    mask_blur = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)

    # 将光晕效果叠加到原图上
    result = cv2.addWeighted(image, 1.0, mask_blur, dot_intensity, 0)

    return result

def add_trapezoid_laser(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, use_gaussian_blur):
    """
    在图片的中间生成梯形激光图样，并添加光晕效果
    :param image_path: 输入图片的路径
    :param dot_intensity: 光点的亮度增强倍数
    :param glow_size_range: 光晕大小的范围（最小值，最大值）
    :param center_brightness: 光斑中心的亮度（0-255）
    :param r: 光点的红色通道值（0-255）
    :param g: 光点的绿色通道值（0-255）
    :param b: 光点的蓝色通道值（0-255）
    :return: 添加激光图样后的图片
    """
    # 加载图片
    image = cv2.imread(image_path)
    if image is None:
        print("错误：无法加载图片，请检查路径是否正确。")
        return None

    # 获取图片尺寸
    height, width = image.shape[:2]

    # 创建一个纯黑色的掩码图像
    mask = np.zeros_like(image)

    # 随机生成梯形的参数
    min_side = min(height, width)
    trapezoid_width = random.randint(min_side // 3, min_side // 1.2)  # 梯形的底边宽度
    trapezoid_height = random.randint(min_side // 3, min_side // 1.2)  # 梯形的高度
    trapezoid_shift = random.randint(-trapezoid_width // 4, trapezoid_width // 4)  # 梯形的倾斜程度

    # 随机生成梯形的位置
    center_x = random.randint(trapezoid_width // 2, width - trapezoid_width // 2)
    center_y = random.randint(trapezoid_height // 2, height - trapezoid_height // 2)

    # 计算梯形的四个顶点
    top_left = (center_x - trapezoid_width // 2 + trapezoid_shift, center_y - trapezoid_height // 2)
    top_right = (center_x + trapezoid_width // 2 + trapezoid_shift, center_y - trapezoid_height // 2)
    bottom_right = (center_x + trapezoid_width // 2, center_y + trapezoid_height // 2)
    bottom_left = (center_x - trapezoid_width // 2, center_y + trapezoid_height // 2)

    # 将顶点转换为 NumPy 数组
    trapezoid_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

    # 在掩码上绘制梯形
    center_color = (int(b * center_brightness / 255), int(g * center_brightness / 255), int(r * center_brightness / 255))
    cv2.fillPoly(mask, [trapezoid_points], center_color)  # 绘制梯形

    if use_gaussian_blur:
        # 随机生成光晕大小（必须是正奇数）
        glow_size = random.randint(glow_size_range[0], glow_size_range[1])
        if glow_size % 2 == 0:  # 确保是奇数
            glow_size += 1

        # 对掩码进行高斯模糊，模拟光晕效果
        mask_blur = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)
    else:
        # 不使用高斯模糊，直接使用原始掩码
        mask_blur = mask

    # 将光晕效果叠加到原图上
    result = cv2.addWeighted(image, 1.0, mask_blur, dot_intensity, 0)

    return result
def add_oval_laser(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, use_gaussian_blur=True):
    """
    在图片的中间生成椭圆形激光图样，并添加光晕效果
    :param image_path: 输入图片的路径
    :param dot_intensity: 光点的亮度增强倍数
    :param glow_size_range: 光晕大小的范围（最小值，最大值）
    :param center_brightness: 光斑中心的亮度（0-255）
    :param r: 光点的红色通道值（0-255）
    :param g: 光点的绿色通道值（0-255）
    :param b: 光点的蓝色通道值（0-255）
    :param use_gaussian_blur: 是否使用高斯模糊（默认为 True）
    :return: 添加激光图样后的图片
    """
    # 加载图片
    image = cv2.imread(image_path)
    if image is None:
        print("错误：无法加载图片，请检查路径是否正确。")
        return None

    # 获取图片尺寸
    height, width = image.shape[:2]

    # 创建一个纯黑色的掩码图像
    mask = np.zeros_like(image)

    # 随机生成椭圆的参数
    min_side = min(height, width)
    ellipse_width = random.randint(min_side // 3, min_side // 1.2)  # 椭圆的宽度
    ellipse_height = random.randint(min_side // 3, min_side // 1.2)  # 椭圆的高度
    angle = random.randint(0, 180)  # 椭圆的旋转角度

    # 随机生成椭圆的位置
    center_x = random.randint(ellipse_width // 2, width - ellipse_width // 2)
    center_y = random.randint(ellipse_height // 2, height - ellipse_height // 2)

    # 在随机位置绘制椭圆
    center_color = (int(b * center_brightness / 255), int(g * center_brightness / 255), int(r * center_brightness / 255))
    cv2.ellipse(
        mask,
        (center_x, center_y),  # 椭圆中心
        (ellipse_width // 2, ellipse_height // 2),  # 椭圆的长轴和短轴
        angle,  # 旋转角度
        0,  # 起始角度
        360,  # 结束角度
        center_color,  # 颜色
        -1  # 填充椭圆
    )

    # 如果使用高斯模糊
    if use_gaussian_blur:
        # 随机生成光晕大小（必须是正奇数）
        glow_size = random.randint(glow_size_range[0], glow_size_range[1])
        if glow_size % 2 == 0:  # 确保是奇数
            glow_size += 1

        # 对掩码进行高斯模糊，模拟光晕效果
        mask_blur = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)
    else:
        # 不使用高斯模糊，直接使用原始掩码
        mask_blur = mask

    # 将光晕效果叠加到原图上
    result = cv2.addWeighted(image, 1.0, mask_blur, dot_intensity, 0)

    return result

def generate(num,image_path,dot_intensity,glow_size_range,center_brightness,r,g,b,type_light,flag):
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
            parts = image_path.split("_")  # 按 "_" 分割
            number = parts[1].split(".")[0]  # 提取 "0" 并去掉 ".jpg"
            cv2.imwrite("jiguang\\{}\\{}_{}_{}.jpg".format(type_light,number, type_light, i), output_image)
            i += 1
        else:
            print("错误：生成图片失败。")
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
            parts = image_path.split("_")  # 按 "_" 分割
            number = parts[1].split(".")[0]  # 提取 "0" 并去掉 ".jpg"
            cv2.imwrite("jiguang\\{}\\{}_{}_{}.jpg".format(type_light,number, type_light, i), output_image)
            i += 1
        else:
            print("错误：生成图片失败。")
            break

def generate_0(image_path, num, glow_size_range,center_brightness):

    type_light = 1
    while type_light <= 8:

        if type_light == 1:
            r, g, b = 217, 58, 54  # 光点的颜色（黄色）
            dot_intensity = 5
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light,flag=True)
        elif type_light == 2:
            r, g, b = 76, 179, 57  # 光点的颜色（黄色）
            dot_intensity = 5
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light,flag=True)
        elif type_light == 3:
            r, g, b = 36, 75, 152  # 光点的颜色（黄色）
            dot_intensity = 5
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light,flag=True)
        elif type_light == 4:
            r= g = b = 80
            dot_intensity = 2.3  # 光晕的亮度增强倍数
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light,flag=False)
        elif type_light == 5:
            r= g = b = 80
            dot_intensity = 2.5  # 光晕的亮度增强倍数

            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light,flag=False)
        elif type_light == 6:
            r= g = b = 80
            dot_intensity = 2.7  # 光晕的亮度增强倍数

            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light,flag=False)
        elif type_light == 7:
            r= g = b = 80
            dot_intensity = 3  # 光晕的亮度增强倍数

            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light,flag=False)
        elif type_light == 8:
            r= g = b = 80
            dot_intensity = 3.5  # 光晕的亮度增强倍数
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light,flag=False)

        type_light += 1