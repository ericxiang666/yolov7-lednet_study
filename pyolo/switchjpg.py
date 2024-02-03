#要換編譯選項
from PIL import Image
import os

def resize_images(folder_path, target_size):
    files = os.listdir(folder_path)
    
    for file in files:
        image_path = os.path.join(folder_path, file)
        try:
            with Image.open(image_path) as image:
                resized_image = image.resize(target_size, Image.ANTIALIAS)
                resized_image.save(image_path)
                print(f"Resized {file}")
        except:
            print(f"Failed to resize {file}")

# 指定資料夾路徑和目標尺寸
folder_path = "D:/pytools/1"
target_size = (640, 640)

# 執行圖片壓縮
resize_images(folder_path, target_size)