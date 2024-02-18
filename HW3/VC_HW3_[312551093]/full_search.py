import cv2
import numpy as np
import time

def motion_estimation(current_frame, reference_frame, block_size, search_range):
    height, width = current_frame.shape
    vectors = np.zeros((height // block_size, width // block_size, 2), dtype=int)
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            current_block = current_frame[i:i + block_size, j:j + block_size]
            min_sad = float('inf')
            for k in range(-search_range, search_range + 1):
                for l in range(-search_range, search_range + 1):
                    ref_block = reference_frame[i + k:i + k + block_size, j + l:j + l + block_size]
                    if ref_block.shape != (block_size, block_size):
                        continue
                    sad = np.sum(np.abs(current_block.astype(int) - ref_block.astype(int)))
                    if sad < min_sad:
                        min_sad = sad
                        vectors[i // block_size, j // block_size] = [k, l]
    return vectors

def motion_compensation(reference_frame, motion_vectors, block_size):
    height, width = reference_frame.shape
    compensated_frame = np.zeros_like(reference_frame)
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            dy, dx = motion_vectors[i // block_size, j // block_size]
            compensated_block = reference_frame[i + dy:i + dy + block_size, j + dx:j + dx + block_size]
            compensated_frame[i:i + block_size, j:j + block_size] = compensated_block
    return compensated_frame

def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


current_frame = cv2.imread('two_gray.png', cv2.IMREAD_GRAYSCALE)
reference_frame = cv2.imread('one_gray.png', cv2.IMREAD_GRAYSCALE)

start_time = time.time()
search_range = 8
motion_vectors = motion_estimation(current_frame, reference_frame, 8, search_range)
compensated_frame = motion_compensation(reference_frame, motion_vectors, 8)
residual = current_frame.astype(int) - compensated_frame.astype(int)
residual = np.clip(residual, 0, 255).astype(np.uint8)

runtime = time.time() - start_time
psnr = calculate_psnr(current_frame, compensated_frame)


cv2.imwrite('reconstructed_frame.png', compensated_frame)
cv2.imwrite('residual.png', residual)

print(f'PSNR: {psnr}')
print(f'Runtime: {runtime} seconds')
