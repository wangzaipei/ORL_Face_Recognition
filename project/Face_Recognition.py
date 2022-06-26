import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
np.random.seed(2022)  # 随机种子

validation_split = 0.3    # 表示拆分数据30% 做测试集
#定义一个图片生成器，加载以及增强图片
image_generator = ImageDataGenerator(rescale=1. / 255., validation_split=validation_split)
# 图片尺寸
im_height = 112
im_width = 92
#图片路径
image_path='data'
#数据块大小
batch_size=128
# 训练集数据生成器，one-hot编码，注意，比方法1多了一个参数subset
train_data_gen = image_generator.flow_from_directory(directory=image_path,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     target_size=(im_height, im_width),
                                                     class_mode='categorical',
                                                     subset='training')
# 测试集数据生成器，one-hot编码
valid_data_gen = image_generator.flow_from_directory(directory=image_path,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     target_size=(im_height, im_width),
                                                     class_mode='categorical',
                                                     subset='validation')

