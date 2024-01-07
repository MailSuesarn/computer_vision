# Author: Suesarn Wilainuch

# Evaluation
from utils import load_and_preprocess
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np


x_train, y_train_U, x_test, y_test_U, y_train, y_test = load_and_preprocess('UC_merced_64x64.h5')

model = tf.keras.models.load_model('model_v1')

# ➖➖➖➖➖➖➖➖➖ TEST ➖➖➖➖➖➖➖➖➖
# Flat predicted test
Attention_layer = tf.keras.Model(model.input, model.get_layer("Attention_layer").output)
Attention_map = Attention_layer.predict(x_test)        # (6720, 64, 64, 4)
arg_Attention_map = np.argmax(Attention_map, axis=-1)  # (6720, 64, 64)
arg_Attention_map_flat = arg_Attention_map.ravel()

# Flat y_test_U
arg_y = np.argmax(y_test_U, axis=-1)
arg_y = arg_y.ravel()

_, acc = model.evaluate(x_test, y_test, verbose=1)
print('Test acc image-level: %.4f%%' % (acc * 100))

acc_pixel = accuracy_score(arg_y, arg_Attention_map_flat)
print('Test acc pixel-level: %.4f%%' % (acc_pixel * 100))

kappa = cohen_kappa_score(arg_Attention_map_flat, arg_y)
print("Kappa score test:", kappa)

CM = confusion_matrix(arg_y, arg_Attention_map_flat)
print("confusion matrix test:\n", CM)

target_names = ['land', 'impervious', 'vegetation', 'water']
print("Pixel level: \n", classification_report(arg_y, arg_Attention_map_flat, target_names=target_names))

