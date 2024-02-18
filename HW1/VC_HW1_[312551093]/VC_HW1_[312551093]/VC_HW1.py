from PIL import Image
import numpy as np

img = Image.open('lena.png')
img_rgb = np.array(img)

R = img_rgb[:,:,0]
G = img_rgb[:,:,1]
B = img_rgb[:,:,2]

Y = 0.299*R + 0.587*G + 0.114*B
U = -0.169*R - 0.331*G + 0.5*B + 128
V = 0.5*R - 0.419*G - 0.081*B + 128

Cb = 128 - 0.168736*R - 0.331264*G + 0.5*B
Cr = 128 + 0.5*R - 0.418688*G - 0.081312*B

Y = np.clip(Y, 0, 255).astype(np.uint8)
U = np.clip(U, 0, 255).astype(np.uint8)
V = np.clip(V, 0, 255).astype(np.uint8)
Cb = np.clip(Cb, 0, 255).astype(np.uint8)
Cr = np.clip(Cr, 0, 255).astype(np.uint8)

channels = [R, G, B, Y, U, V, Cb, Cr]
channel_names = ['R', 'G', 'B', 'Y', 'U', 'V', 'Cb', 'Cr']

for channel, name in zip(channels, channel_names):
    img_channel = Image.fromarray(channel) 
    img_channel.show(title=name)
    img_channel.save(f"{name}.png")
