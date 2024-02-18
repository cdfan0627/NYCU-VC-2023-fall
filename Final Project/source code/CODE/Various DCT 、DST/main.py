#%%
import glob
import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct, dst, idst
from tqdm import tqdm
from utils import jpeg_quantize, jpeg_dequantize, zigzag, inverse_zigzag, RLE_encode, RLE_decode, calculate_PSNR
from multiprocessing import Pool, RLock, freeze_support

def func(variant_type, block_size):
    root = r"C:\Users\steven\Downloads\vimeo_test_clean\sequences"

    # List all paths of the images
    folder_paths = glob.glob(os.path.join(root, "*"))
    img_paths = []
    for folder_path in folder_paths:
        clip_paths = glob.glob(os.path.join(folder_path, "*"))
        for clip_path in clip_paths:
            img_paths_ = glob.glob(os.path.join(clip_path, "*"))
            img_paths += img_paths_

    img_sizes = []
    img_PSNRs = []
    for img_path in tqdm(img_paths, position=variant_type, desc=f"variant type: {variant_type}"):
        block_seq = []
        img = np.int32(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
        img = np.int8(img-128)
        for y in range(0, img.shape[0], block_size):
            for x in range(0, img.shape[1], block_size):
                block_freq_img = dst(dst(img[y:y+block_size, x:x+block_size], norm="ortho", type=variant_type).T, norm="ortho", type=variant_type).T
                block_freq_img = jpeg_quantize(block_freq_img)
                seq = np.int8(zigzag(block_freq_img))
                block_seq.append(RLE_encode(seq))

        img_ = np.zeros_like(img)
        for y in range(int(img.shape[0]//block_size)):
            for x in range(int(img.shape[1]//block_size)):
                seq = RLE_decode(block_seq[y*int(img.shape[1]//block_size)+x])
                block_freq_img = inverse_zigzag(seq, block_size)
                block_freq_img = jpeg_dequantize(block_freq_img)
                img_[y*block_size:(y+1)*block_size, x*block_size:(x+1)*block_size] = np.int8(idst(idst(block_freq_img, norm="ortho", type=variant_type).T, norm="ortho", type=variant_type).T)
        
        img_sizes.append(np.int8(np.concatenate(block_seq)).nbytes)
        img_PSNRs.append(calculate_PSNR(img, img_))

    print(f"Variant type: {variant_type}, average bytes: {np.mean(np.array(img_sizes))} (bytes/image)")
    print(f"Variant type: {variant_type}, average PSNR: {np.mean(np.array(img_PSNRs))} (PSNR/image)")



if __name__ == "__main__":
    root = r"C:\Users\steven\Downloads\vimeo_test_clean\sequences"
    block_size = 8


    # For multi-processing
    # freeze_support()
    # p = Pool(4, initializer=tqdm.set_lock, initargs=(RLock(),))
    # for i in range(1, 5):
    #     p.apply_async(func, (i,block_size))
    #     time.sleep(3)
    # p.close()
    # p.join()

    variant_type = 2
    print(variant_type)
    print("="*50)

    # List all paths of the images
    folder_paths = glob.glob(os.path.join(root, "*"))
    img_paths = []
    for folder_path in folder_paths:
        clip_paths = glob.glob(os.path.join(folder_path, "*"))
        for clip_path in clip_paths:
            img_paths_ = glob.glob(os.path.join(clip_path, "*"))
            img_paths += img_paths_

    img_sizes = []
    img_PSNRs = []
    for img_path in tqdm(img_paths, position=variant_type, desc=f"variant type: {variant_type}"):
        block_seq = []
        img = np.int32(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
        img = np.int8(img-128)
        print(img.nbytes)
    #     for y in range(0, img.shape[0], block_size):
    #         for x in range(0, img.shape[1], block_size):
    #             block_freq_img = dct(dct(img[y:y+block_size, x:x+block_size], norm="ortho", type=variant_type).T, norm="ortho", type=variant_type).T
    #             block_freq_img = jpeg_quantize(block_freq_img)
    #             seq = np.int8(zigzag(block_freq_img))
    #             block_seq.append(RLE_encode(seq))

    #     img_ = np.zeros_like(img)
    #     for y in range(int(img.shape[0]//block_size)):
    #         for x in range(int(img.shape[1]//block_size)):
    #             seq = RLE_decode(block_seq[y*int(img.shape[1]//block_size)+x])
    #             block_freq_img = inverse_zigzag(seq, block_size)
    #             block_freq_img = jpeg_dequantize(block_freq_img)
    #             img_[y*block_size:(y+1)*block_size, x*block_size:(x+1)*block_size] = np.int8(idct(idct(block_freq_img, norm="ortho", type=variant_type).T, norm="ortho", type=variant_type).T)
        
    #     img_sizes.append(np.int8(np.concatenate(block_seq)).nbytes)
    #     img_PSNRs.append(calculate_PSNR(img, img_))

    # print(f"Variant type: {variant_type}, average bytes: {np.mean(np.array(img_sizes))} (bytes/image)")
    # print(f"Variant type: {variant_type}, average PSNR: {np.mean(np.array(img_PSNRs))} (PSNR/image)")
    
    # plt.figure()
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    # ax1.imshow(img)
    # ax2.imshow(img_)
    # ax1.title.set_text('Original image')
    # ax2.title.set_text('After coding')
    # ax1.axis("off")
    # ax2.axis("off")
    # plt.show()


# %%
