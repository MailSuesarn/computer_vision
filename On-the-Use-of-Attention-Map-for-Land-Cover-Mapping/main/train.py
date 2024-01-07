# Author: Suesarn Wilainuch

from utils import load_and_preprocess
from utils import select_supervised_samples
from model import define_attention_model
import numpy as np
import tensorflow as tf

# hyperparameter
total_data = 26880
N_percent = 100
IMAGE_SHAPE = (64, 64, 3)
NUM_CLASSES = 4
LEARNING_RATE = 0.001
N_SAMPLE_LABEL = int(total_data * (N_percent / 100.0))
BATCH_SIZE = 64
EPOCH = 1

x_train, y_train_U, x_test, y_test_U, y_train, y_test = load_and_preprocess('UC_merced_64x64.h5')

data = [x_train, y_train]

dataset_supervised = select_supervised_samples(data, n_samples=N_SAMPLE_LABEL)
imgs, labels = dataset_supervised
print(imgs.shape, labels.shape)

model = define_attention_model(image_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE)

class_weight = labels.shape[0] / (NUM_CLASSES * labels.sum(axis=0))
class_weight_dict = dict(enumerate(class_weight))

model.fit(imgs, labels, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1, class_weight=class_weight_dict)

filename = 'model_v1'
model.save(filename)