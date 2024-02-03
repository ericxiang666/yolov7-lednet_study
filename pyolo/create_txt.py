import os

def generate_jpg_paths_txt(folder_path, txt_file_path):
    with open(txt_file_path, 'w') as f:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.jpg'):
                    file_path = os.path.join(root, file)
                    f.write(file_path + '\n')

# 指定文件夹路径和生成的txt文件路径
folder_path = r"/content/drive/MyDrive/yolov7c/photography/images/train"
txt_file_path = r"/content/drive/MyDrive/yolov7c/photography/train.txt"

# 生成txt文件
generate_jpg_paths_txt(folder_path, txt_file_path)
