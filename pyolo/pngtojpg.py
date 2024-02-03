from PIL import Image
import os

def png_to_jpg(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有PNG文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # 构造输入和输出文件的完整路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".jpg")

            # 打开PNG图像
            img = Image.open(input_path)

            # 转换并保存为JPEG
            img = img.convert("RGB")
            img.save(output_path, "JPEG")

if __name__ == "__main__":
    # 指定输入和输出文件夹
    input_folder = "D:/all"
    output_folder = "D:/pytools/2"

    # 执行转换
    png_to_jpg(input_folder, output_folder)
