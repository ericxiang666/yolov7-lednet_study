import PIL.Image as Image
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import warnings
from torchvision import transforms
from torchvision.transforms import functional as TF
warnings.filterwarnings("ignore")
imagepath ='D:/Aerial photography/train (69).jpg'

# read image with PIL module
img_pil = Image.open(imagepath, mode='r')
img_pil = img_pil.convert('RGB')
# img_pil

transform = transforms.Compose([
    #transforms.Resize((480,480)),
    transforms.RandomHorizontalFlip(p=0.9),
])

new_img = transform(img_pil)
# new_img

new_img.save('D:/re/image69.jpg')


# plt.imshow(np.array(new_img))
# plt.show()