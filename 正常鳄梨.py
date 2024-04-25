import numpy as np
import matplotlib.pyplot as plt
import os

def map_minmax(x, Il, Ih, It, ht):
    return (x - Il) / (Ih - Il) * (ht - It) + It

def dist(v1, v2, compress=(1.0, 1.0, 1.0)):
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    xc, yc, zc = compress
    dst2 = ((x1 - x2) * xc) ** 2 + ((y1 - y2) * yc) ** 2 + ((z1 - z2) * zc) ** 2
    return dst2 ** 0.5

# Initialize the 3D array
a = np.zeros([100, 100, 150])

# Core points
c1 = (50, 50, 50)
c2 = (50, 50, 110)

# Processing the data
for x in range(100):
    for y in range(100):
        for z in range(150):
            n = np.random.normal(scale=0.25)
            if dist((x, y, z), c1) <= 21:
                a[x, y, z] = 1
            elif z <= 60 and dist((x, y, z), c1, (1.0, 1.0, 1.1)) < 40 + n:
                a[x, y, z] = 2
            elif z >= 115 and dist((x, y, z), c2, (1.0, 1.0, 1.1)) < 17 + n:
                a[x, y, z] = 2
            elif 60 < z < 115:
                r = map_minmax(z, 60, 115, 38.7, 16.2)
                if dist((x, y, 0), (50, 50, 0)) < r + n:
                    a[x, y, z] = 2
            if a[x, y, z] == 0:
                n = np.random.normal(scale=0.4)
                if z <= 60 and dist((x, y, z), c1, (1.0, 1.0, 1.1)) < 47 + n:
                    a[x, y, z] = 3
                if z >= 115 and dist((x, y, z), c2, (1.0, 1.0, 1.1)) < 24 + n:
                    a[x, y, z] = 3
                if 60 < z < 115:
                    r = map_minmax(z, 60, 115, 45.9, 23.4)
                    if dist((x, y, 0), (50, 50, 0)) < r + n:
                        a[x, y, z] = 3

# Displaying slices from three different directions: x, y, and z
# Displaying slices from three different directions: x, y, and z
plt.figure(figsize=(12, 4))

# Slice along the x-axis (transpose for vertical display)
plt.subplot(1, 3, 1)
plt.imshow(np.transpose(a[50, :, :]), cmap='gray')
plt.title('Vertical Slice along X-axis (x=50)')

# Slice along the y-axis (transpose for vertical display)
plt.subplot(1, 3, 2)
plt.imshow(np.transpose(a[:, 50, :]), cmap='gray')
plt.title('Vertical Slice along Y-axis (y=50)')

# Slice along the z-axis (no need to transpose)
plt.subplot(1, 3, 3)
plt.imshow(a[:, :, 75], cmap='gray')
plt.title('Slice along Z-axis (z=75)')

plt.tight_layout()
plt.show()
# 添加的代码，用于将切片保存为PGM文件
def save_pgm(image, filename):
    """保存一个二维数组到PGM文件"""
    height, width = image.shape
    max_val = 255  # PGM格式的最大像素值

    # 检查图像最大值是否为零，避免除以零
    max_image_val = image.max()
    if max_image_val > 0:
        normalized_image = (image / max_image_val) * max_val
    else:
        normalized_image = np.zeros_like(image)

    # 确保没有 NaN 或无限值
    normalized_image = np.nan_to_num(normalized_image)

    # 转换为整数类型
    normalized_image = normalized_image.astype(int)

    # 创建PGM文件头
    header = f'P2\n{width} {height}\n{max_val}\n'

    # 写入文件
    with open(filename, 'w') as f:
        f.write(header)
        for row in normalized_image:
            row_str = ' '.join(str(val) for val in row)
            f.write(row_str + '\n')

# X方向切片（转置）
for i in range(a.shape[0]):
    slice_x = np.transpose(a[i, :, :])  # 转置切片
    save_pgm(slice_x, f'healthy_x_{i}.pgm')

# Y方向切片（转置）
for i in range(a.shape[1]):
    slice_y = np.transpose(a[:, i, :])  # 转置切片
    save_pgm(slice_y, f'healthy_y_{i}.pgm')

# Z方向切片（不需要转置）
for i in range(a.shape[2]):
    slice_z = a[:, :, i]
    save_pgm(slice_z, f'healthy_z_{i}.pgm')




