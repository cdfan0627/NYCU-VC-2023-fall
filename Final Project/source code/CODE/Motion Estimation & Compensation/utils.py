import numpy as np
import copy

def jpeg_quantize(freq_img):
    table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]])
    quantized_img = np.around(freq_img / table)
    return quantized_img

def jpeg_dequantize(freq_img):
    table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]])
    quantized_img = np.around(freq_img * table)
    return quantized_img


def zigzag(img):
    x, y = 0, 0
    height, width = img.shape
    output_seq = []

    while x < width and y < height:
        output_seq.append(img[y, x])

        if (x+y)%2 == 0:
            if y == 0:
                if x == width:
                    y += 1
                else:
                    x += 1

            elif x == width-1 and y < height:
                y += 1
            
            elif y > 0 and x < width-1:
                y -= 1
                x += 1

        else:
            if y == height-1 and x < width-1:
                x += 1
            
            elif x == 0:
                if y == height-1:
                    x += 1
                else:
                    y += 1
            elif y < height-1 and x > 0:
                y += 1
                x -= 1

        if x == width-1 and y == height-1:
            break

    output_seq.append(img[y, x])
    return np.array(output_seq)

def inverse_zigzag(input, block_size):
    x = 0
    y = 0
    i = 0
    output = np.zeros((block_size, block_size), dtype=np.float32)
    while x < block_size and y < block_size: 	
		# go upper right
        if (x + y) % 2 == 0:
            if y == 0:
                output[y, x] = input[i]
                if x == block_size - 1:
                    y += 1
                else:
                    x += 1
                i += 1
            elif x == block_size - 1  and y < block_size:
                output[y, x] = input[i]
                y += 1
                i += 1
            else:
                output[y, x] = input[i]
                y -= 1
                x += 1
                i += 1
        # go lower left
        else:
            if x == 0:
                output[y, x] = input[i]
                if y == block_size - 1:
                    x += 1
                else:
                    y += 1
                i += 1
            elif y == block_size - 1 and x < block_size:
                output[y, x] = input[i]
                x += 1
                i += 1
            else:
                output[y, x] = input[i]
                y += 1
                x -= 1
                i += 1
    return output

def RLE_encode(img, bits=8):
    encoded = []
    count = 0
    prev = None

    for pixel in img:
        if prev==None:
            prev = pixel
            count+=1
        else:
            if prev!=pixel:
                encoded.append((count, prev))
                prev=pixel
                count=1
            else:
                if count<(2**bits)-1:
                    count+=1
                else:
                    encoded.append((count, prev))
                    prev=pixel
                    count=1
    encoded.append((count, prev))
   
    return np.array(encoded)

def RLE_decode(encoded):
    decoded=[]
    for rl in encoded:
        r,p = rl[0], rl[1]
        decoded.extend([p]*int(r))
    return decoded

def calculate_PSNR(ref, cur):
    ref = np.float32(ref)+128.0001
    cur = np.float32(cur)+128
    # for i in (ref-cur).flatten():
    #     print(i)
    rmse = np.sqrt(np.mean((ref-cur)**2))
    if rmse <= 1e-6:
        rmse = 1e-6
    if np.max(ref) <= 0:
        print("np.max(ref) <= 0", np.max(ref))
    return 20*np.log10(np.max(ref)/rmse)

def MSE(block1, block2):
    error = np.mean((block1 - block2)**2)
    return error

def MAD(block1, block2):
    error = np.mean(abs(block1 - block2))
    return error

def center(x, y, block_size):
    return int(x + block_size / 2), int(y + block_size / 2)

def select_area(x, y, previous_img, block_size, search_range):
    h, w = previous_img.shape
    cx, cy = center(x, y, block_size)
    sx1 = max(0, cx - int(block_size / 2) - search_range)
    sx2 = min(sx1 + search_range*2 + block_size, w)
    sy1 = max(0, cy - int(block_size / 2) - search_range)
    sy2 = min(sy1 + search_range*2 + block_size, h)
    search_area = previous_img[sy1:sy2, sx1:sx2]
    return search_area

def get_block(cx, cy, search_area, block_size):
    bx, by = cx - int(block_size / 2), cy - int(block_size / 2)
    bx, by = max(0, bx), max(0, by)
    previous_block = search_area[by:by + block_size, bx:bx + block_size]
    return previous_block, bx, by

def nine_point(cx, cy, step):
        p1 = (cx, cy)
        p2 = (cx + step, cy)
        p3 = (cx, cy + step)
        p4 = (cx + step, cy + step)
        p5 = (cx - step, cy)
        p6 = (cx, cy - step)
        p7 = (cx - step, cy - step)
        p8 = (cx + step, cy - step)
        p9 = (cx - step, cy + step)
        plist = [p1,p2,p3,p4,p5,p6,p7,p8,p9]
        return plist

def block_match(current_block, search_area, block_size, search_range, mode = 'full', eval = 'MAD'):
    if mode == 'full':
        ah, aw = search_area.shape
        min_error = float("+inf")
        min_p = None
        for cx in range(int(block_size/2), int(aw - block_size/2)):
            for cy in range(int(block_size/2), int(ah - block_size/2)):
                previous_block, bx, by = get_block(cx, cy, search_area, block_size)
                if eval == 'MAD':
                    error = MAD(current_block, previous_block)
                elif eval == 'MSE':
                    error = MSE(current_block, previous_block)
                else:
                    print('Evaluation must be MAD or MSE.')

                if error <= min_error:
                    min_error = error
                    min_p = (bx, by)

        bx, by = min_p
        match_block = search_area[by:by + block_size, bx:bx + block_size]
        return match_block
    elif mode == '3step':
        step = int(block_size / 2 + search_range / 2)
        ah, aw = search_area.shape
        acy, acx = int(ah / 2), int(aw / 2)
        plist = nine_point(acx, acy, step)
        min_error = float("+inf")
        min_p = None
        while step >= 1:
            for p in range(len(plist)):
                cx, cy = plist[p]
                previous_block, bx, by = get_block(cx, cy, search_area, block_size)
                if eval == 'MAD':
                    error = MAD(current_block, previous_block)
                elif eval == 'MSE':
                    error = MSE(current_block, previous_block)
                else:
                    print('Evaluation must be MAD or MSE.')
                if error <= min_error:
                    min_error = error
                    min_p = (bx, by)
            step = int(step / 2)
            plist = nine_point(bx, by, step)
        bx, by = min_p
        match_block = search_area[by:by + block_size, bx:bx + block_size]
        return match_block
    else:
        print('Mode must be full or 3step.')

def motion_compnesation(previous_img, current_img, block_size, search_range, mode = 'full', eval = 'MAD'):
    h, w = previous_img.shape
    predict_img = np.ones((h, w))*255
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            current_block = current_img[y:y + block_size, x:x + block_size]
            search_area = select_area(x, y, previous_img, block_size, search_range) 
            match_block = block_match(current_block, search_area, block_size, search_range, mode = 'full', eval = 'MAD')
            predict_img[y:y + block_size, x:x + block_size] = match_block
    return predict_img