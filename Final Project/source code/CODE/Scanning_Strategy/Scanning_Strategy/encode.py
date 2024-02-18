import torch
import numpy as np
import os

from CustomDataset import VCDataset, VCDataset2
from torch.utils.data import DataLoader

import utils
import tqdm

batch_size = 4
save_dir = '/media/oislab/E_disk/VC/strategy_fig/test9'
train_residual = False
table_quantize = False

if train_residual:
    train_dataset = VCDataset2('/media/oislab/E_disk/VC/vimeo_test_clean/test_sequneces', None)
else:
    train_dataset = VCDataset('/media/oislab/E_disk/VC/vimeo_test_clean/test_sequneces', None)
train_loader = DataLoader(train_dataset, batch_size=batch_size)

zero_counts = torch.zeros((8, 8))
rle_length1 = []
rle_length2 = []
rle_length3 = []
rle_length4 = []
rle_length_zigzag = []
strategy2 = utils.handcraft_strategy1
strategy3 = utils.handcraft_strategy2
strategy4 = utils.nn_strategy

s_name2 = 'handcraft1'
s_name3 = 'handcraft2'
s_name4 = 'nn'
for data in tqdm.tqdm(train_loader):
    img = data.squeeze(1)
    if train_residual:
        quantized_dct_coefs = utils.dct_8x8(utils.block_splitting(img)).to(torch.int16).to(torch.float32)
        strategy1 = utils.count_zero_strategy_res
        s_name1 = 'count zero res'

    else:
        if table_quantize:
            quantized_dct_coefs = (torch.round(utils.dct_8x8(utils.block_splitting(img))/torch.from_numpy(utils.y_table)))
            strategy1 = utils.count_zero_strategy_ori_q
            s_name1 = 'count zero ori q'
        else:
            quantized_dct_coefs = utils.dct_8x8(utils.block_splitting(img)).to(torch.int16).to(torch.float32)
            strategy1 = utils.count_zero_strategy_ori_noq
            s_name1 = 'count zero ori noq'

    quantized_dct_coefs_numpy = np.array(quantized_dct_coefs, dtype=np.int16)

    # print(quantized_dct_coefs_numpy[:, 2, 0])
    batch, h, w = quantized_dct_coefs_numpy.shape

    for quantized_dct_coef in quantized_dct_coefs_numpy:
        rle_data1 = utils.run_length_encode2(quantized_dct_coef, strategy1)
        rle_data2 = utils.run_length_encode2(quantized_dct_coef, strategy2)
        rle_data3 = utils.run_length_encode2(quantized_dct_coef, strategy3)
        rle_data4 = utils.run_length_encode2(quantized_dct_coef, strategy4)
        rle_data_zigzag = utils.run_length_encode2(quantized_dct_coef, utils.zigzag_strategy)
        
        rle_length1.append(len(rle_data1))
        rle_length2.append(len(rle_data2))
        rle_length3.append(len(rle_data3))
        rle_length4.append(len(rle_data4))
        rle_length_zigzag.append(len(rle_data_zigzag))
    
    # break

print(f'{s_name1}:', np.mean(rle_length1), np.median(rle_length1), np.std(rle_length1))
print(f'{s_name2}:', np.mean(rle_length2), np.median(rle_length2), np.std(rle_length2))
print(f'{s_name3}:', np.mean(rle_length3), np.median(rle_length3), np.std(rle_length3))
print(f'{s_name4}:', np.mean(rle_length4), np.median(rle_length4), np.std(rle_length4))
print(f'zigzag:', np.mean(rle_length_zigzag), np.median(rle_length_zigzag), np.std(rle_length_zigzag))

    

# print(zero_counts)
# print(utils.make_strategy_from_zero_count(zero_counts))