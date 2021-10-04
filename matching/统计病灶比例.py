import os
import PIL.Image as Image
import cv2
import numpy as np
from tqdm import tqdm



def get_ct_ratio(img):
    l = cv2.imread(os.path.join(CT_PATH, img), cv2.IMREAD_GRAYSCALE)
    # print(img)
    ill = (l > 0).astype(np.uint8)
    ill_ratio = ill.sum()
    return ill_ratio

def get_mri_ratio(img):
    l = cv2.imread(os.path.join(MRI_PATH, img), cv2.IMREAD_GRAYSCALE)
    # print(img)
    l1 = (l > 25).astype(np.uint8)
    l2 = (l < 75).astype(np.uint8)
    label = np.logical_and(l1, l2).astype(np.uint8)
    ill_ratio = label.sum()
    return ill_ratio

if __name__ == "__main__":

    MRI_PATH=r'D:\Project\CycleGAN-and-pix2pix-master\datasets\mri2ct\mask'
    CT_PATH=r'D:\Project\CycleGAN-and-pix2pix-master\datasets\mri2ct\trianb2zhihua'
    MRI_ratio_dict = {}
    CT_ratio_dict = {}
    for img in tqdm(os.listdir(MRI_PATH)):
        ratio=get_mri_ratio(os.path.join(MRI_PATH,img))
        MRI_ratio_dict[img]=ratio

    for img in tqdm(os.listdir(CT_PATH)):
        ratio = get_ct_ratio(os.path.join(CT_PATH, img))
        CT_ratio_dict[img] = ratio


    for mri in tqdm(MRI_ratio_dict):
        mri_ratio=MRI_ratio_dict[mri]
        ratio_distance_dict={}
        for img in CT_ratio_dict:
            ct_ratio=CT_ratio_dict[img]
            distance=abs(ct_ratio-mri_ratio)
            ratio_distance_dict[img]=distance
        np.save(os.path.join(r'D:\Project\blog_code-master\blog_code-master\img_sim_hash\ratio',mri+'_ratio.npy'), ratio_distance_dict)