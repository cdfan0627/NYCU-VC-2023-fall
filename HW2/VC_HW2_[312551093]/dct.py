import numpy as np
import cv2
import time


def dct2D(matrix):
    M, N = matrix.shape
    dct_result = np.zeros((M, N))
    alpha = np.array([1.0/np.sqrt(2.0) if i == 0 else 1.0 for i in range(M)])
    beta = np.array([1.0/np.sqrt(2.0) if j == 0 else 1.0 for j in range(N)])

    for u in range(M):
        for v in range(N):
            cos_i = np.cos(np.pi * (2 * np.arange(M) + 1) * u / (2 * M))
            cos_j = np.cos(np.pi * (2 * np.arange(N) + 1) * v / (2 * N))
            dct_result[u, v] = alpha[u] * beta[v] * np.sum(matrix * cos_i[:, None] * cos_j)

    return (2.0 / np.sqrt(M * N)) * dct_result

def idct2D(matrix):
    M, N = matrix.shape
    img_result = np.zeros((M, N))
    alpha = np.array([1.0/np.sqrt(2.0) if i == 0 else 1.0 for i in range(M)])
    beta = np.array([1.0/np.sqrt(2.0) if j == 0 else 1.0 for j in range(N)])

    for i in range(M):
        for j in range(N):
            cos_u = np.cos(np.pi * (2 * i + 1) * np.arange(M) / (2 * M))
            cos_v = np.cos(np.pi * (2 * j + 1) * np.arange(N) / (2 * N))
            img_result[i, j] = np.sum(alpha * beta * matrix * cos_u[:, None] * cos_v)

    return (2.0 / np.sqrt(M * N)) * img_result

def dct1D(signal):
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.sqrt(2/N) * np.cos(np.pi * (2*n+1) * k / (2*N))
    M[0, :] /= np.sqrt(2)
    return np.dot(M, signal)

def two_1d_dct(image):
    dct_rows = np.apply_along_axis(dct1D, axis=1, arr=image)
    dct_cols = np.apply_along_axis(dct1D, axis=0, arr=dct_rows)
    return dct_cols

def idct_1d(signal):
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.sqrt(2/N) * np.cos(np.pi * (2*n+1) * k / (2*N))
    M[:, 0] /= np.sqrt(2)
    return np.dot(signal, M)

def two_1d_idct(dct_coefficients):
    idct_rows = np.apply_along_axis(idct_1d, axis=1, arr=dct_coefficients)
    idct_cols = np.apply_along_axis(idct_1d, axis=0, arr=idct_rows)
    return idct_cols

def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    max_pixel_value = 255.0
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))


img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

start_time = time.time()
dct_coeffs = dct2D(img)
scaled_dct_coefficients = np.log1p(np.abs(dct_coeffs))
scaled_dct_coefficients = (scaled_dct_coefficients / np.max(scaled_dct_coefficients) * 255).astype(np.uint8)
cv2.imwrite("2D_DCT_Coefficients.png", scaled_dct_coefficients)
reconstructed = idct2D(dct_coeffs)
cv2.imwrite("2D_IDCT_reconstructed.png", reconstructed)
end_time = time.time()
print(f"2D-DCT Runtime: {end_time - start_time} seconds")
image_psnr = psnr(img, reconstructed)
print(f"2D-DCT PSNR: {image_psnr} dB")



start_time = time.time()
two_1d_dct_coefficients = two_1d_dct(img)
scaled_two_1d_dct_coefficients = np.log1p(np.abs(two_1d_dct_coefficients))
scaled_two_1d_dct_coefficients = (scaled_two_1d_dct_coefficients / np.max(scaled_two_1d_dct_coefficients) * 255).astype(np.uint8)
cv2.imwrite("two_1D_DCT_Coefficients.png", scaled_two_1d_dct_coefficients)
two_1d_reconstructed = two_1d_idct(two_1d_dct_coefficients).clip(0, 255).astype(np.uint8)
cv2.imwrite("two_1D_IDCT_reconstructed.png", two_1d_reconstructed)
end_time = time.time()
print(f"two 1D-DCT Runtime: {end_time - start_time} seconds")
two_1d_dct_psnr = psnr(img, two_1d_reconstructed)
print(f"two 1D-DCT PSNR: {two_1d_dct_psnr} dB")




