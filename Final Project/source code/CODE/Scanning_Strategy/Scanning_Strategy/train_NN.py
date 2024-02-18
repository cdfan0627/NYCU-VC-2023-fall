import torch
import numpy as np
import os


from Model import StrategyModel
from CustomDataset import VCDataset, VCDataset2
from torch.utils.data import DataLoader
import utils

ITERATION = 10000
batch_size = 8
learning_rate = 1e-4
crop_size = 16
save_dir = '/media/oislab/E_disk/VC/strategy_fig/test15'
train_residual = False
table_quantize = False

n_head = 8
layer_channels = [1, 32, 64, 32, 1]

model = StrategyModel(n_head=n_head, layer_channels=layer_channels).to('cuda')
if train_residual:
    train_dataset = VCDataset2('/media/oislab/E_disk/VC/vimeo_test_clean/sequences', crop_size)
else:
    train_dataset = VCDataset('/media/oislab/E_disk/VC/vimeo_test_clean/sequences', crop_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(0.9, 0.999))

iteration = 0

utils.make_dir(save_dir)

while iteration < ITERATION:
    for data in train_loader:
        iteration += 1
        optimizer.zero_grad()
        
        img = data.squeeze(1)
        if train_residual:
            quantized_dct_coefs = utils.dct_8x8(utils.block_splitting(img)).to(torch.int16).to(torch.float32).unsqueeze(1).to('cuda')
        else:
            if table_quantize:
                quantized_dct_coefs = (utils.diff_round(utils.dct_8x8(utils.block_splitting(img))/utils.y_table_torch)).unsqueeze(1).to('cuda')
            else:
                quantized_dct_coefs = utils.dct_8x8(utils.block_splitting(img)).to(torch.int16).to(torch.float32).unsqueeze(1).to('cuda')
        
        strategy = model(quantized_dct_coefs)
        
        quantized_dct_coefs_numpy = np.array(quantized_dct_coefs.detach().cpu().squeeze(1), dtype=np.int16)
        batch, h, w = quantized_dct_coefs_numpy.shape

        rle_encoded_batch_coded_len = []
        for quantized_dct_coef in quantized_dct_coefs_numpy:
            rle_data = utils.run_length_encode2(quantized_dct_coef, strategy)
            
            rle_encoded_batch_coded_len.append(len(rle_data))

        rle_encoded_batch_coded_len = torch.tensor(rle_encoded_batch_coded_len, dtype=torch.float32)
        
        l = torch.tensor(-torch.log2(1/(rle_encoded_batch_coded_len+1e-6)), requires_grad=True)

        torch.mean(l).backward()
        optimizer.step()


        with open(os.path.join(save_dir, 'loss.txt'), 'a') as f:
            f.write(f'iteration: bpp:{torch.mean(rle_encoded_batch_coded_len):.3f} \n')
        
        if iteration % 100 == 0:
            utils.plot_strategy(strategy, iteration, save_dir)
            with open(os.path.join(save_dir, 'strategy.txt'), 'a') as f:
                f.write(f'{iteration}: {strategy} \n')
                
            