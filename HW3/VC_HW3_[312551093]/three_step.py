import cv2
import numpy as np
import time

def three_step_search(current_frame, reference_frame, block_size):
    height, width = current_frame.shape
    vectors = np.zeros((height // block_size, width // block_size, 2), dtype=int)
    step_size = 4
    while step_size >= 1:
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                current_block = current_frame[i:i + block_size, j:j + block_size]
                min_sad = float('inf')
                best_vector = [0, 0]
                for k in range(-step_size, step_size + 1, step_size):
                    for l in range(-step_size, step_size + 1, step_size):
                        ref_block_x = i + vectors[i // block_size, j // block_size][0] + k
                        ref_block_y = j + vectors[i // block_size, j // block_size][1] + l
                        if ref_block_x < 0 or ref_block_x + block_size > height or ref_block_y < 0 or ref_block_y + block_size > width:
                            continue
                        ref_block = reference_frame[ref_block_x:ref_block_x + block_size, ref_block_y:ref_block_y + block_size]
                        sad = np.sum(np.abs(current_block.astype(int) - ref_block.astype(int)))
                        if sad < min_sad:
                            min_sad = sad
                            best_vector = [vectors[i // block_size, j // block_size][0] + k, vectors[i // block_size, j // block_size][1] + l]
                vectors[i // block_size, j // block_size] = best_vector
        step_size //= 2
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

motion_vectors = three_step_search(current_frame, reference_frame, 8)
compensated_frame = motion_compensation(reference_frame, motion_vectors, 8)
residual = current_frame.astype(int) - compensated_frame.astype(int)
residual = np.clip(residual, 0, 255).astype(np.uint8)

runtime = time.time() - start_time
psnr = calculate_psnr(current_frame, compensated_frame)


cv2.imwrite('reconstructed_frame.png', compensated_frame)
cv2.imwrite('residual.png', residual)

print(f'PSNR: {psnr}')
print(f'Runtime: {runtime} seconds')