#%%
import glob
import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct, dst, idst
from tqdm import tqdm
from utils import *
from multiprocessing import Pool, RLock, freeze_support

def func(variant_type, block_size, search_range, mode, eval):
    path = r"C:\Users\enerson\Downloads\sequences"
    img_sizes = []
    img_PSNRs = []
    
    for m in tqdm(os.listdir(path), position=variant_type, desc=f"variant type: {variant_type}"):
        for n in os.listdir(os.path.join(path, m)):
            img1 = np.int32(cv2.imread(os.path.join(path, m , n, os.listdir(os.path.join(path, m , n))[0]), cv2.IMREAD_GRAYSCALE))
            for img_path in os.listdir(os.path.join(path, m , n))[1:]:
                img2 = np.int32(cv2.imread(os.path.join(path, m , n, img_path), cv2.IMREAD_GRAYSCALE))
                predict_img = motion_compnesation(img1, img2, block_size, search_range, mode, eval)
                residual = np.int8(predict_img - img1)
                # residual = np.int8(img1 - img2)
                block_seq = []
                for y in range(0, img1.shape[0], block_size):
                    for x in range(0, img1.shape[1], block_size):
                        block_freq_img = dct(dct(residual[y:y+block_size, x:x+block_size], norm="ortho", type=variant_type).T, norm="ortho", type=variant_type).T
                        block_freq_img = jpeg_quantize(block_freq_img)
                        seq = np.int8(zigzag(block_freq_img))
                        block_seq.append(RLE_encode(seq))

                img_ = np.zeros_like(img1)
                for y in range(int(img1.shape[0]//block_size)):
                    for x in range(int(img1.shape[1]//block_size)):
                        seq = RLE_decode(block_seq[y*int(img1.shape[1]//block_size)+x])
                        block_freq_img = inverse_zigzag(seq, block_size)
                        block_freq_img = jpeg_dequantize(block_freq_img)
                        img_[y*block_size:(y+1)*block_size, x*block_size:(x+1)*block_size] = np.int8(idct(idct(block_freq_img, norm="ortho", type=variant_type).T, norm="ortho", type=variant_type).T)
                # plt.figure()
                # plt.subplot(1,2,1)
                # plt.imshow(img2, cmap='gray')
                # plt.subplot(1,2,2)
                # plt.imshow(np.uint8(img1+img_), cmap='gray')
                # plt.show()
                img_sizes.append(np.int8(np.concatenate(block_seq)).nbytes)
                img_PSNRs.append(calculate_PSNR(residual, img_))
                img1 = np.int32(cv2.imread(os.path.join(path, m , n, img_path), cv2.IMREAD_GRAYSCALE))
                


    print(f"Variant type: {variant_type}, average bytes: {np.mean(np.array(img_sizes))} (bytes/image)")
    print(f"Variant type: {variant_type}, average PSNR: {np.mean(np.array(img_PSNRs))} (PSNR/image)")



if __name__ == "__main__":
    block_size = 8
    search_range = 8
    mode = '3step' # full or 3step
    eval = 'MAD' # MAD or MSE


    # For multi-processing
    freeze_support()
    p = Pool(4, initializer=tqdm.set_lock, initargs=(RLock(),))
    for i in range(1, 5):
        p.apply_async(func, (i,block_size, search_range, mode, eval))
        time.sleep(3)
    p.close()
    p.join()
