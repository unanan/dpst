import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageTransform():
    def __init__(self, size=224):
        self.transform = transforms.Compose([
            # TODO: Add Transforms
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def __call__(self, image):
        if len(image.split())!=3:
            image=image.convert('RGB')

        return self.transform(image)


class BinaryDataset(Dataset):
    def __init__(self,img_folder,img_size):
        super(BinaryDataset, self).__init__()
        self.img_list = glob.glob(os.path.join(img_folder,"*.*"))
        self.transform = ImageTransform(img_size)


    def __getitem__(self, item):
        imagename = os.path.split(self.img_list[item])[1]

        image = Image.open(self.img_list[item]).convert('RGB')
        #TODO add image process
        image = None

    def __len__(self):
        return len(self.img_list)