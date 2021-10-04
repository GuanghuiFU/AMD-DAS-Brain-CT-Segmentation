import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch


class MyPipeiCtDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.dir_A = opt.dataroot + '/' + opt.phase + 'A/'
        self.dir_B = opt.dataroot + '/' + opt.phase + 'B/'
        self.dir_mask='D:\Project\CycleGAN-and-pix2pix-master\datasets\mri2ct\mask/'


        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.mask_paths = sorted(make_dataset(self.dir_mask, opt.max_dataset_size))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_Mask = get_transform(self.opt, grayscale=(input_nc == 1),convert=False)
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    @classmethod
    def processmask(cls, type,pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'

        pil_img=np.array(pil_img)
        if type=='MRI':
            l1 = (pil_img > 25).astype(np.uint8)  # 大于25的是1，小于25的是0
            # img=Image.fromarray(l1)
            l1=np.expand_dims(l1, 0)
            return torch.from_numpy(l1)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        # print('mri--', A_path[25:])
        mri_name=A_path[25:]
        read_dictionary = np.load(
            'D:\Project\\blog_code-master\\blog_code-master\img_sim_hash\mix\\'+mri_name+'_mix.npy',
            allow_pickle=True).item()
        n = 20
        L = sorted(read_dictionary.items(), key=lambda item: item[1], reverse=False)
        L = L[:n]
        dictdata = {}
        for l in L:
            dictdata[l[0]] = l[1]
        ct_list=list(dictdata.keys())
        ct_path= random.choice(ct_list)
        print(ct_path)
        mask_path=self.mask_paths[index % self.A_size]
        print('mask',mask_path)
        B_path =os.path.join(r'D:\Project\CycleGAN-and-pix2pix-master\datasets\mri2ct\trainB',ct_path)

        A_img = Image.open(A_path).convert('RGB')
        mask=Image.open(mask_path)
        mask_img = self.processmask(pil_img=mask, type='MRI', scale=1)
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        M=mask_img
        B = self.transform_B(B_img)

        return {'A': A, 'B': B,'Mask':M, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
