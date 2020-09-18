import os
import glob
import numpy as np
import cv2
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms

class ImageTransform():
    def __init__(self, size=512):
        self.transform = transforms.Compose([
            # TODO: Add Transforms
            # transforms.Resize((size, size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def __call__(self, image):
        if len(image.split())!=3:
            image=image.convert('RGB')

        return self.transform(image)


class BinaryDataset(Dataset):
    def __init__(self,img_size):
        super(BinaryDataset, self).__init__()
        self.transform = ImageTransform(img_size)
        self.img_size = img_size

        self.max_circlenum = 3

    def __randradius(self,expt=None):
        if expt:
            return random.choice([i for i in range(4, (self.img_size>>1)-1) if i!=expt])
        else:
            return random.randint(4, (self.img_size>>1)-1)

    def __randcenter(self,radius,expt=None):
        if expt:
            return random.choice([i for i in range(radius,self.img_size-radius) if i!=expt])
        else:
            return random.randint(radius,self.img_size-radius)

    def __randthick(self,radius,expt=None):
        if expt:
            return min(random.choice([i for i in range(1, 10) if i!=expt]),radius>>1)
        else:
            return min(random.randint(1, 10),radius>>1)



    def __getitem__(self, item):
        circlenum = random.randint(1,self.max_circlenum)
        images = [np.zeros((self.img_size,self.img_size,3),np.uint8),
                  np.zeros((self.img_size,self.img_size,3),np.uint8),
                  np.zeros((self.img_size,self.img_size,3),np.uint8)]

        for _ in range(circlenum):
            radius = self.__randradius()
            center_x = self.__randcenter(radius)
            center_y = self.__randcenter(radius)
            thickness = self.__randthick(radius) # 厚度不变性
            images[0] = cv2.circle(images[0], (center_x,center_y),radius,(1,1,1),thickness)

            thickness = self.__randthick(radius)
            images[1] = cv2.circle(images[1], (center_x, center_y), radius, (1, 1, 1), thickness)

            rec1 = np.random.randint(2)
            rec2 = np.random.randint(2)
            radius = self.__randradius(expt=radius) if rec1 else radius
            center_x = self.__randcenter(radius,expt=center_x) if rec2 else center_x
            center_y = self.__randcenter(radius,expt=center_y) if np.random.randint(2) or (1-rec1-rec2)>0 else center_y
            images[2] = cv2.circle(images[2], (center_x, center_y), radius, (1, 1, 1), thickness)

        for id,_ in enumerate(images):
            images[id] = self.transform(Image.fromarray(cv2.cvtColor(images[id],cv2.COLOR_BGR2RGB)))

        # pos,pos,neg
        return images, circlenum


    def __len__(self):
        #TODO
        return 1#len(self.img_list)