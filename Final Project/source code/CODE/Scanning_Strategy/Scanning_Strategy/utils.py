
import itertools
import numpy as np
import torch
import torch.nn as nn
import heapq
from collections import Counter
import matplotlib.pyplot as plt
import os


def block_splitting(image):
    """ Splitting image into patches
    Input:
        image(tensor): batch x height x width
    Output: 
        patch(tensor):  batch x h*w/64 x h x w
    """
    k = 8
    batch_size, height = image.shape[0:2]
    image_reshaped = image.view(batch_size, height // k, k, -1, k)
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    return image_transposed.contiguous().view(-1, k, k)

def dct_8x8(image):
    """ Discrete Cosine Transformation
    Input:
        image(tensor): batch x height x width
    Output:
        dcp(tensor): batch x height x width
    """
    image = image - 128
    tensor = torch.zeros((8, 8, 8, 8), dtype=torch.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
            (2 * y + 1) * v * np.pi / 16)
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7, dtype=np.float32)
    scale = torch.outer(torch.from_numpy(alpha), torch.from_numpy(alpha)) * 0.25
    result = scale * torch.tensordot(image, tensor, dims=2)
    result.view(image.shape)
    return result

def idct_8x8(image):
    """ Inverse discrete Cosine Transformation
    Input:
        dcp(tensor): batch x height x width
    Output:
        image(tensor): batch x height x width
    """
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7, dtype=np.float32)
    alpha = torch.outer(torch.from_numpy(alpha), torch.from_numpy(alpha))
    image = image * alpha

    tensor = torch.zeros((8, 8, 8, 8), dtype=torch.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
            (2 * v + 1) * y * np.pi / 16)

    result = 0.25 * torch.tensordot(image, tensor, dims=2) + 128
    result.view(image.shape)
    return result

def block_merging(patches, batch_size, height, width):
    """ Merge pathces into image
    Inputs:
        patches(tensor) batch x height*width/64, height x width
        height(int)
        width(int)
    Output:
        image(tensor): batch x height x width
    """
    k = 8
    image_reshaped = patches.view(batch_size, height//k, width//k, k, k)
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    return image_transposed.contiguous().view(batch_size, height, width)

def diff_round(x):
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    return torch.round(x) - x.detach() + x

y_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float32).T

y_table_torch = nn.Parameter(torch.from_numpy(y_table))

updown_strategy = [0,15,16,31,32,47,48,63,1,14,17,30,33,46,49,62,2,13,18,29,34,45,50,61,3,12,19,28,35,44,51,60,4,11,20,27,36,43,52,59,5,10,21,26,37,42,53,58,6,9,22,25,38,41,54,57,7,8,23,24,39,40,55,56]

def run_length_encode(block, scan_strategy):
    """
    Apply Run-Length Encoding (RLE) to an 8x8 block of DCT coefficients using a custom scan strategy.
    
    :param block: 8x8 block of DCT coefficients.
    :param scan_strategy: List of indices representing the scanning order.
    :return: The RLE encoded sequence.
    """
    # Flatten the block according to the custom scan strategy

    flattened = block.flatten()[scan_strategy]
    # Apply RLE
    rle_sequence = []
    zero_count = 0
    for coeff in flattened:
        if coeff == 0:
            zero_count += 1
        else:
            # Append the count of zeros followed by the non-zero coefficient
            rle_sequence.append((zero_count, coeff))
            zero_count = 0
    
    # End-of-Block (EOB) symbol, typically (0, 0)
    rle_sequence.append((0, 0))

    return rle_sequence

def run_length_encode2(img, scan_strategy):
    """
    img: Grayscale img.
    """
    
    encoded = []
    count = 0
    prev = None
    fimg = img.flatten()[scan_strategy]
    for pixel in fimg:
        if prev==None:
            prev = pixel
            count+=1
        else:
            if prev!=pixel:
                encoded.append((count, prev))
                prev=pixel
                count=1
            else:
                count+=1
    encoded.append((count, prev))
   
    return encoded

def discretize_and_normalize(bpp_tensor, min_bpp, max_bpp, num_bins):
    """
    Discretizes and normalizes bpp values into a distribution.

    :param bpp_tensor: A 1D tensor of bpp values with shape (batch size, ).
    :param min_bpp: The minimum expected bpp value.
    :param max_bpp: The maximum expected bpp value.
    :param num_bins: The number of bins to use for discretization.
    :return: A 2D tensor representing the normalized distribution of bpp values.
    """
    # Create bins for bpp values
    bins = torch.linspace(min_bpp, max_bpp, steps=num_bins)

    # Digitize the bpp values into bins
    bpp_indices = torch.bucketize(bpp_tensor, bins, right=True)

    # Create a histogram of bpp values
    bpp_histogram = torch.zeros(bpp_tensor.size(0), num_bins)
    bpp_histogram.scatter_add_(1, bpp_indices.unsqueeze(1) - 1, torch.ones_like(bpp_tensor).unsqueeze(1))

    # Normalize the histogram to form a probability distribution
    normalized_bpp_distribution = bpp_histogram / bpp_histogram.sum(dim=1, keepdim=True)

    return normalized_bpp_distribution


class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    # For priority queue to compare nodes
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(symbols_frequency):
    priority_queue = [HuffmanNode(sym, freq) for sym, freq in symbols_frequency.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)

        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(priority_queue, merged)

    return priority_queue[0]

def assign_codes(node, prefix="", codebook={}):
    if node is not None:
        if node.symbol is not None:
            codebook[node.symbol] = prefix
        assign_codes(node.left, prefix + "0", codebook)
        assign_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_coding(data):
    # Frequency analysis for Huffman coding
    frequency = Counter(data)
    
    # Building the Huffman tree
    root = build_huffman_tree(frequency)
    
    # Generate Huffman codes
    huffman_codes = assign_codes(root)

    # Encode the RLE data with Huffman codes
    encoded_data = ''
    for symbol in data:
        try:
            encoded_data += huffman_codes[symbol]
        except KeyError as e:
            print(f"KeyError - no Huffman code for symbol: {e}")
            # Handle the missing code or break the loop, depending on your needs.
            # For instance, you might break, return, or raise a more informative error.
            break

    return encoded_data, huffman_codes


def get_size_of_encoded_data(dct_coefs, strategy):
    rle_data = run_length_encode(dct_coefs, strategy)
    total_rle_data_points = len(rle_data)

    _, huffman_table = huffman_coding(rle_data)
    total_huffman_code_length = sum(len(code) for code in huffman_table.values())

    return total_rle_data_points, total_huffman_code_length

# Example usage with a flattened block of coefficients after RLE
# rle_data = [(0, -34), (3, 2), (2, 3), (1, 1)]  # Your actual RLE data
# encoded_data, codes = huffman_coding(rle_data)
# print("Huffman Codes:", codes)
# print("Encoded Data:", encoded_data)


def plot_strategy(strategy, iteration, save_dir):
    # Define the 8x8 block size
    block_size = 8

    # Define the zigzag path or replace this with your strategy path

    # Generate a 2D grid to represent the block
    grid = np.zeros((block_size, block_size))

    # Draw the path: Increment grid values along the path to show the order
    for order, z in enumerate(strategy):
        grid[z//8, z%8] = order + 1  # start numbering from 1

    # Plotting
    plt.figure(1)
    fig, ax = plt.subplots()
    cax = ax.matshow(grid, cmap="viridis")
    plt.title("Strategy Path on 8x8 Block")
    plt.colorbar(cax)

    # Annotate the path order on the grid
    for (i, j), val in np.ndenumerate(grid):
        ax.text(j, i, int(val), ha='center', va='center', color='red')\

    plt.savefig(os.path.join(save_dir, f'{iteration}.png'))
    plt.close(1)

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def make_strategy_from_zero_count(zero_counts):
    # zero_counts_np = zero_counts.numpy()

    sorted_indices = np.argsort(zero_counts, axis=None)

    sorted_2d_indices = [(i // 8, i % 8) for i in sorted_indices]
    flattened_strategy = [i * 8 + j for i, j in sorted_2d_indices]

    return flattened_strategy


ori_zero_count_noq = [[   74036., 10064449., 19121572., 24116616., 31138816., 37282088.,
         44531972., 51407188.],
        [ 9634236., 17690378., 24136872., 30135556., 35593032., 42119156.,
         48192024., 54578420.],
        [18418378., 23749994., 28945652., 34266424., 39072184., 45081040.,
         50735636., 56465360.],
        [22901312., 29291250., 33798384., 38421872., 42620952., 48059480.,
         53250800., 58371232.],
        [29830164., 34626096., 38507076., 42574304., 44855480., 51174888.,
         55870360., 60430808.],
        [35129960., 40375908., 43752564., 47210048., 50325880., 54724364.,
         58852680., 62820860.],
        [42119928., 46194800., 49117368., 52085524., 54657924., 58520256.,
         62038012., 65375408.],
        [48772628., 52214916., 54426424., 56829052., 58823332., 62150980.,
         65071568., 67797984.]]
ori_zero_count_q = [[   74036., 10064449., 19121582., 24116602., 31138812., 37282088.,
         44531980., 51407156.],
        [ 9634236., 17690376., 24136872., 30135552., 35593044., 42119112.,
         48192036., 54578408.],
        [18418392., 23750016., 28945648., 34266432., 39072144., 45081012.,
         50735624., 56465336.],
        [22901296., 29291256., 33798400., 38421824., 42620964., 48059504.,
         53250784., 58371280.],
        [29830156., 34626072., 38507084., 42574304., 44855536., 51174884.,
         55870364., 60430784.],
        [35129984., 40375908., 43752576., 47210092., 50325892., 54724364.,
         58852672., 62820896.],
        [42119924., 46194784., 49117344., 52085528., 54657916., 58520196.,
         62038076., 65375344.],
        [48772644., 52214956., 54426400., 56829072., 58823352., 62150956.,
         65071556., 67798088.]]

count_zero_strategy_ori_noq = [0, 8, 1, 9, 16, 2, 24, 17, 3, 10, 18, 25, 32, 11, 4, 26, 19, 33, 40, 12, 5, 27, 34, 20, 41, 13, 48, 35, 28, 42, 6, 36, 21, 49, 43, 29, 14, 56, 50, 44, 22, 37, 7, 51, 57, 30, 58, 15, 52, 45, 38, 23, 59, 31, 53, 60, 46, 39, 54, 61, 47, 62, 55, 63]
count_zero_strategy_ori_q =   [0, 8, 1, 9, 16, 2, 24, 17, 3, 10, 18, 25, 32, 11, 4, 26, 19, 33, 40, 12, 5, 27, 34, 20, 41, 13, 48, 35, 28, 42, 6, 36, 21, 49, 43, 29, 14, 56, 50, 44, 22, 37, 7, 51, 57, 30, 58, 15, 52, 45, 38, 23, 59, 31, 53, 60, 46, 39, 54, 61, 47, 62, 55, 63]
zigzag_strategy =             [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]
handcraft_strategy1 = [0, 1, 8, 9, 2, 16, 10, 17, 18, 3, 24, 11, 25, 19, 26, 27, 4, 32, 12, 33, 20, 34, 28, 35, 36, 5, 40, 13, 41, 21, 42, 29, 43, 37, 44, 45, 6, 48, 14, 49, 22, 50, 30, 51, 38, 52, 46, 53, 54, 7, 56, 15, 57, 23, 58, 31, 59, 39, 60, 47, 61, 55, 62, 63]
handcraft_strategy2 = [0, 1, 8, 9, 2, 16, 10, 17, 18, 3, 24, 11, 25, 19, 26, 4, 32, 12, 33, 27, 20, 34, 5, 28, 35, 40, 13, 41, 21, 42, 36, 29, 43, 6, 48, 14, 49, 22, 50, 37, 44, 30, 51, 7, 56, 15, 45, 57, 38, 52, 23, 58, 31, 59, 46, 53, 39, 60, 54, 47, 61, 55, 62, 63]
nn_strategy = [8, 1, 0, 2, 9, 16, 3, 17, 4, 10, 26, 40, 24, 51, 11, 7, 27, 57, 6, 18, 50, 31, 32, 14, 44, 20, 60, 54, 21, 5, 61, 33, 45, 62, 22, 12, 38, 19, 15, 48, 36, 52, 35, 29, 49, 43, 34, 25, 30, 28, 41, 42, 39, 46, 55, 47, 37, 23, 13, 58, 53, 56, 63, 59]
if __name__ == '__main__':
    
    plot_strategy(count_zero_strategy_ori_noq, 'count_zero_strategy_ori_noq', '/media/oislab/E_disk/VC/strategy_fig')
    plot_strategy(count_zero_strategy_ori_q, 'count_zero_strategy_ori_q', '/media/oislab/E_disk/VC/strategy_fig')
    # grid = a

    # Plotting
    # plt.figure(1)
    # fig, ax = plt.subplots()
    # cax = ax.matshow(grid, cmap="viridis")
    # plt.title("Zero count on training data")
    # plt.colorbar(cax)

    # # Annotate the path order on the grid
    # for (i, j), val in np.ndenumerate(grid):
    #     ax.text(j, i, int(val), ha='center', va='center', color='red')\

    # plt.savefig(os.path.join('/media/oislab/E_disk/VC/strategy_fig', f'{iteration}.png'))
    # plt.close(1)
    # plot_strategy(handcraft_strategy1, 'handcraft1_strategy', '/media/oislab/E_disk/VC/strategy_fig')
    # plot_strategy(handcraft_strategy2, 'handcraft2_strategy', '/media/oislab/E_disk/VC/strategy_fig')
    # print(np.array(c) > np.array(d))

    # import cv2

    # img1 = cv2.imread('/media/oislab/E_disk/VC/vimeo_test_clean/sequences/00001/0266/im2.png', 0) 
    # img2 = cv2.imread('/media/oislab/E_disk/VC/vimeo_test_clean/sequences/00001/0266/im3.png', 0)

    # h, w = img1.shape

    # img1 = torch.tensor(img1, dtype=torch.float32).unsqueeze(0)
    # img2 = torch.tensor(img2, dtype=torch.float32).unsqueeze(0)



    # x = img1


    # x_blocks = block_splitting(x)
    # dct_coefs = dct_8x8(x_blocks)

    # dct_coefs_numpy = dct_coefs.to(torch.int16).numpy()

    # e1 = []
    # e2 = []
    # i = 0

    # for dct_coef in dct_coefs_numpy:
    #     i += 1
        
    #     rle_data1 = sparsity_loss(dct_coef, zigzag_strategy)
    #     # encoded_data1, table1 = huffman_coding(rle_data1)
    #     rle_data2 = sparsity_loss(dct_coef, updown_strategy)
    #     # encoded_data2, table2 = huffman_coding(rle_data2)

    #     e1.append(rle_data1)
    #     e2.append(rle_data2)
    #     break
            

    # print(e1)
    # print(e2)

    