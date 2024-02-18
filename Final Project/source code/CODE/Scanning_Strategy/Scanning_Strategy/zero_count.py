import torch
import numpy as np
import os

from CustomDataset import VCDataset, VCDataset2
from torch.utils.data import DataLoader

import utils
import tqdm

batch_size = 64
save_dir = '/media/oislab/E_disk/VC/strategy_fig/test9'
train_residual = False
table_quantize = False

if train_residual:
    train_dataset = VCDataset2('/media/oislab/E_disk/VC/vimeo_test_clean/sequences', None)
else:
    train_dataset = VCDataset('/media/oislab/E_disk/VC/vimeo_test_clean/sequences', None)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

zero_counts = torch.zeros((8, 8))
pone_counts = torch.zeros((8, 8))
none_counts = torch.zeros((8, 8))
for data in tqdm.tqdm(train_loader):
    img = data.squeeze(1)
    if train_residual:
        quantized_dct_coefs = utils.dct_8x8(utils.block_splitting(img)).to(torch.int16).to(torch.float32)
    else:
        if table_quantize:
            quantized_dct_coefs = (torch.round(utils.dct_8x8(utils.block_splitting(img))/torch.from_numpy(utils.y_table)))
        else:
            quantized_dct_coefs = utils.dct_8x8(utils.block_splitting(img)).to(torch.int16).to(torch.float32)
    zero_counts += (quantized_dct_coefs == 0).sum(dim=0)

print(zero_counts)