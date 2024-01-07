# Author: Suesarn Wilainuch

# Visualization Attention map and segmented image
from utils import load_and_preprocess
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# Load dataset
x_train, y_train_U, x_test, y_test_U, y_train, y_test = load_and_preprocess('UC_merced_64x64.h5')

# Load model
model = tf.keras.models.load_model('model_v1')

# encode class
land_index = 0
man_made_index = 1
plant_index = 2
water_index = 3

# RGB represent color map
land = (145, 89, 60)
man_made = (128, 128, 128)
plant = (0, 255, 0)
water = (0, 255, 255)

arg_y = np.argmax(y_test_U, axis=-1)  # (6720, 64, 64 ,4) --> (6720, 64, 64)
Attention_layer = tf.keras.Model(model.input, model.get_layer("Attention_layer").output)
Attention_map = Attention_layer.predict(x_test)  # (6720, 64, 64, 4)
arg_Attention_map = np.argmax(Attention_map, axis=-1)  # (6720, 64, 64)

# select 1 sample
order = 24

AM = Attention_map[order]  # (64, 64, 4)
input_img = x_test[order]  # (64, 64 ,3)
index_label = arg_y[order]  # (64, 64)
index_Attention_map = arg_Attention_map[order]  # (64, 64)

# convert index to RGB
RGB_label = Image.new('RGB', (64, 64), 0x000000)
w, h = RGB_label.size
for x in range(h):
    for y in range(w):
        index_class = index_label[y][x]  # 0, 1, 2, 3

        if index_class == land_index:
            RGB_label.putpixel((x, y), land)

        elif index_class == man_made_index:
            RGB_label.putpixel((x, y), man_made)

        elif index_class == plant_index:
            RGB_label.putpixel((x, y), plant)

        elif index_class == water_index:
            RGB_label.putpixel((x, y), water)

# segmented image in RGB mode
segment = Image.new('RGB', (64, 64), 0x000000)
w, h = segment.size
for x in range(h):
    for y in range(w):
        index_class = index_Attention_map[y][x]  # 0, 1, 2, 3

        if index_class == land_index:
            segment.putpixel((x, y), land)

        elif index_class == man_made_index:
            segment.putpixel((x, y), man_made)

        elif index_class == plant_index:
            segment.putpixel((x, y), plant)

        elif index_class == water_index:
            segment.putpixel((x, y), water)

plt.axis('off')
plt.imshow(AM[:, :, 0], cmap='jet', alpha=1)
filename = 'Attention_map_land.png'
plt.savefig(filename)
plt.close()

plt.axis('off')
plt.imshow(AM[:, :, 1], cmap='jet', alpha=1)
filename = 'Attention_map_man_made.png'
plt.savefig(filename)
plt.close()

plt.axis('off')
plt.imshow(AM[:, :, 2], cmap='jet', alpha=1)
filename = 'Attention_map_plant.png'
plt.savefig(filename)
plt.close()

plt.axis('off')
plt.imshow(AM[:, :, 3], cmap='jet', alpha=1)
filename = 'Attention_map_water.png'
plt.savefig(filename)
plt.close()

plt.axis('off')
plt.imshow(input_img)
filename = 'input_image.png'
plt.savefig(filename)
plt.close()

plt.axis('off')
plt.imshow(RGB_label)
filename = 'RGB_label.png'
plt.savefig(filename)
plt.close()

plt.axis('off')
plt.imshow(segment)
filename = 'segmented.png'
plt.savefig(filename)
plt.close()
