from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
import os
import numpy as np
from PIL import Image
import torch
from augmentations import get_composed_augmentations
import random



torch.manual_seed(7)
np.random.seed(7)
random.seed(7)

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, type,scale=1):
        self.type=type
        self.imgs_dir = imgs_dir #图像路径
        self.masks_dir = masks_dir #标签路径
        self.scale = scale #裁剪比例
        self.patch_threshold = 0.05  # 0.05
        self.data_rate = 0.66  # 0.66
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.imgs_path=[]
        self.imgs_path1=[]  #取肿瘤像素点小于阈值的数据
        self.imgs_path2=[]  #取肿瘤像素点大于阈值的数据
        self.label_path1=[]
        self.label_path2=[]

        if self.type == 'MRI':
            for folder_name in os.listdir(self.imgs_dir):
                # print(folder_name)
                self.imgs_path += glob.glob(os.path.join(self.imgs_dir, folder_name, "*.png"))
        if self.type == 'CT':
            for folder_name in os.listdir(self.imgs_dir):
                # print(folder_name)
                self.imgs_path += glob.glob(os.path.join(self.imgs_dir, folder_name, "*.png"))
        # print('imglen', len(self.imgs_path))

        self.label_path = []
        if self.type == 'MRI':
            for folder_name in os.listdir(self.masks_dir):
                self.label_path += glob.glob(os.path.join(self.masks_dir, folder_name, "*.jpg"))
        if self.type == 'CT':
            for folder_name in os.listdir(self.masks_dir):
                self.label_path += glob.glob(os.path.join(self.masks_dir, folder_name, "*.png"))



    def __len__(self):
        return len(self.imgs_path)

    @classmethod
    def preprocess(cls,type, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((160, 160))

        img_nd = np.array(pil_img).astype(float)
        # print(img_nd.shape)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        if type=='MRI':
            tf = transforms.Compose([transforms.Normalize([0.086], [0.167])])
        elif type=='CT':
            tf = transforms.Compose([transforms.Normalize([0.074], [0.148])])
        img_trans=torch.from_numpy(img_trans)
        img_trans=tf(img_trans)

        return img_trans

    @classmethod
    def processmask(cls, type,pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'

        pil_img=np.array(pil_img)
        if type=='MRI':
            l1 = (pil_img > 25).astype(np.uint8)  # 大于25的是1，小于25的是0
            img_nd = l1.astype(np.uint8)
            img_trans = np.expand_dims(img_nd, axis=2)
            img_trans = img_trans.transpose((2, 0, 1))
            return img_trans
        if type == 'CT':
            l1 = (pil_img > 0).astype(np.uint8)
            img_nd = l1.astype(np.uint8)
            img_trans = np.expand_dims(img_nd, axis=2)
            img_trans = img_trans.transpose((2, 0, 1))
            return img_trans

    def __getitem__(self, i):


        img_path=self.imgs_path[i]
        label_path=self.label_path[i]
        # print('label_path',label_path)


        mask_file = label_path
        img_file =img_path


        mask = Image.open(mask_file)
        img = Image.open(img_file)

        aug_dict = {'hflip': 0.5, 'vflip': 0.5, 'brightness': 0.2}

        augmentation = get_composed_augmentations(aug_dict=aug_dict)

        img, mask = augmentation(img, mask)

        img = self.preprocess(pil_img=img, type=self.type ,scale=self.scale)
        mask = self.processmask(pil_img=mask,type=self.type ,scale=self.scale)

        return {'image': img, 'mask': torch.from_numpy(mask)}




class TestDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, type,scale=1):
        self.type=type
        self.imgs_dir = imgs_dir #图像路径
        self.masks_dir = masks_dir #标签路径
        self.scale = scale #裁剪比例
        self.patch_threshold = 0.05  # 0.05
        self.data_rate = 0.66  # 0.66
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.imgs_path=[]
        self.imgs_path1=[]  #取肿瘤像素点小于阈值的数据
        self.imgs_path2=[]  #取肿瘤像素点大于阈值的数据
        self.label_path1=[]
        self.label_path2=[]

        if self.type == 'MRI':
            for folder_name in os.listdir(self.imgs_dir):
                # print(folder_name)
                self.imgs_path += glob.glob(os.path.join(self.imgs_dir, folder_name, "*.png"))
        if self.type == 'CT':
            for folder_name in os.listdir(self.imgs_dir):
                # print(folder_name)
                self.imgs_path += glob.glob(os.path.join(self.imgs_dir, folder_name, "*.png"))
        # print('imglen', len(self.imgs_path))

        self.label_path = []
        if self.type == 'MRI':
            for folder_name in os.listdir(self.masks_dir):
                self.label_path += glob.glob(os.path.join(self.masks_dir, folder_name, "*.jpg"))
        if self.type == 'CT':
            for folder_name in os.listdir(self.masks_dir):
                self.label_path += glob.glob(os.path.join(self.masks_dir, folder_name, "*.png"))



    def __len__(self):
        return len(self.imgs_path)

    @classmethod
    def preprocess(cls,type, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((160, 160))

        img_nd = np.array(pil_img).astype(float)
        # print(img_nd.shape)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        if type=='MRI':
            tf = transforms.Compose([transforms.Normalize([0.078], [0.141])])
        elif type=='CT':
            tf = transforms.Compose([transforms.Normalize([0.074], [0.148])])
        img_trans=torch.from_numpy(img_trans)
        img_trans=tf(img_trans)

        return img_trans

    @classmethod
    def processmask(cls, type,pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'

        pil_img=np.array(pil_img)
        if type=='MRI':
            l1 = (pil_img > 25).astype(np.uint8)  # 大于25的是1，小于25的是0
            img_nd = l1.astype(np.uint8)
            img_trans = np.expand_dims(img_nd, axis=2)
            img_trans = img_trans.transpose((2, 0, 1))
            return img_trans
        if type == 'CT':
            l1 = (pil_img > 0).astype(np.uint8)
            img_nd = l1.astype(np.uint8)
            img_trans = np.expand_dims(img_nd, axis=2)
            img_trans = img_trans.transpose((2, 0, 1))
            return img_trans

    def __getitem__(self, i):


        img_path=self.imgs_path[i]
        label_path=self.label_path[i]
        # print('label_path',label_path)


        mask_file = label_path
        img_file =img_path


        mask = Image.open(mask_file)
        img = Image.open(img_file)


        img = self.preprocess(pil_img=img, type=self.type ,scale=self.scale)
        mask = self.processmask(pil_img=mask,type=self.type ,scale=self.scale)

        return {'image': img, 'mask': torch.from_numpy(mask)}






class CTdataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_path = []
        for img in os.listdir(img_dir):
            img = os.path.join(img_dir, img)
            self.img_path.append(img)

    def __len__(self):
        return len(self.img_path)

    @classmethod
    def preprocess(cls, img):
        # img = img.resize((160, 160))
        img_nd = np.array(img).astype(float)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        img_original=img_trans
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        tf = transforms.Compose([transforms.Normalize([0.074], [0.148])])
        img_trans = torch.from_numpy(img_trans)
        img_trans = tf(img_trans)
        return img_trans,img_original

    def __getitem__(self, item):
        # print('ctpath',self.img_path[item])
        img = Image.open(self.img_path[item])
        img,img_original = self.preprocess(img)

        return img,img_original

#源：原始MRI，目标：CT  测试集：500CT脑实质

SOURCE_IMG_DIR=r'D:\Project\Data\data\style_transfer\mixgancu_random3'
SOURCE_LABEL_DIR=r'D:\Project\Data\data\style_transfer\mixgancu_random3_label'
SOURCE_VALI_IMG_DIR=r'D:\Project\Data\data\vali_big_tf_rm\imgs'
SOURCE_VALI_LABEL_DIR=r'D:\Project\Data\data\vali_big\labels'
TARGET_IMG_DIR=r'D:\Project\Data\data\ct240_rm'

TARGET_VALI_IMG_DIR=r'D:\Project\Data\data\naoshizhi_label\Intraparenchymal_hemorrhage_segmentation1\240\cases_remove_bone_images'
TARGET_VALI_LABEL_DIR=r'D:\Project\Data\data\naoshizhi_label\Intraparenchymal_hemorrhage_segmentation1\240\cases_labels'


sourcevalidationset=TestDataset(imgs_dir=SOURCE_VALI_IMG_DIR,masks_dir=SOURCE_VALI_LABEL_DIR,type='MRI')
dataset=BasicDataset(imgs_dir=SOURCE_IMG_DIR,masks_dir=SOURCE_LABEL_DIR,type='MRI')
targetset=CTdataset(img_dir=TARGET_IMG_DIR)
targetvalidationset = TestDataset(imgs_dir=TARGET_VALI_IMG_DIR, masks_dir=TARGET_VALI_LABEL_DIR, type='CT')



train_dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=0)
test_dataloader = DataLoader(sourcevalidationset, batch_size=6, shuffle=True, num_workers=0)
target_dataloader=DataLoader(targetset,batch_size=12,shuffle=True,num_workers=0)
targetvalidationloader = DataLoader(dataset=targetvalidationset, batch_size=12)


if __name__ =='__main__':

    for train_batch in train_dataloader:
        print(train_batch[1])

    for test_batch in test_dataloader:
        print(test_batch)
