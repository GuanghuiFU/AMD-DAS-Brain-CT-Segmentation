import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_phash(s_img_url):
    np_lena_img = cv2.imread(s_img_url, cv2.IMREAD_UNCHANGED)
    np_lena_img_gray = cv2.resize(np_lena_img ,(32, 32), cv2.INTER_AREA)
    np_lena_img_dct = cv2.dct(np.float32(np_lena_img_gray))
    np_lena_img_dct_low_freq = np_lena_img_dct[0:8, 0:8]
    a = np.mean(np_lena_img_dct_low_freq)
    diff = np_lena_img_dct_low_freq > a
    phash_bi = ''.join(str(b) for b in 1 * diff.flatten())
    phash_hex = '{:0>{width}x}'.format(int(phash_bi, 2), width=16)
    return phash_bi, phash_hex

"""calculate hanmming"""
def get_hanming(s_hash_a, s_hash_b):
    if len(s_hash_a) != len(s_hash_b):
        print("two hash dim is not same!")
        return 100000000

    n_hanmming = 0
    for i in range(len(s_hash_a)):
        if s_hash_a[i] != s_hash_b[i]:
            n_hanmming += 1
    return  n_hanmming


if __name__ == "__main__":

    MRI_PATH=r'D:\Project\CycleGAN-and-pix2pix-master\datasets\mri2ct\trainA'
    CT_PATH=r'D:\Project\CycleGAN-and-pix2pix-master\datasets\mri2ct\trainB_original2'
    MRI_hash_dict = {}
    CT_hash_dict = {}
    for img in tqdm(os.listdir(MRI_PATH)):
        phash_bi, phash_hex=get_phash(os.path.join(MRI_PATH,img))
        MRI_hash_dict[img]=phash_bi
    np.save('mri_hash.npy',MRI_hash_dict)
    print('MRI保存完成')
    for img in tqdm(os.listdir(CT_PATH)):
        phash_bi, phash_hex = get_phash(os.path.join(CT_PATH, img))
        CT_hash_dict[img] = phash_bi
    np.save('ct_hash.npy', CT_hash_dict)
    print('CT保存完成')

    for mri in tqdm(MRI_hash_dict):
        mri_hash=MRI_hash_dict[mri]
        hanming_dict={}
        for img in CT_hash_dict:
            ct_hash=CT_hash_dict[img]
            hanming=get_hanming(mri_hash,ct_hash)
            hanming_dict[img]=hanming
        np.save(os.path.join(r'D:\Project\blog_code-master\blog_code-master\img_sim_hash\hanming',mri+'_hanming.npy'), hanming_dict)