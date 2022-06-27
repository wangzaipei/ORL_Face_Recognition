import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,AveragePooling2D

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
batch_size=4
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

nb_filters = 32  # 卷积核的个数
pool_size = (2, 2)  # 池化层的大小
kernel_size = (5, 5)  # 卷积核的大小
input_shape = (im_width, im_height,1)  # 输入图片的维度
nb_classes = 40  # 分类数目

# 构建模型
model = Sequential()
model.add(Conv2D(6,kernel_size,input_shape=input_shape,strides=1))  # 卷积层1
model.add(AveragePooling2D(pool_size=pool_size,strides=2))  # 池化层
model.add(Conv2D(12,kernel_size,strides=1))  # 卷积层2
model.add(AveragePooling2D(pool_size=pool_size,strides=2))  # 池化层
model.add(Flatten())  # 拉成一维数据
model.add(Dense(nb_classes))  # 全连接层2
model.add(Activation('sigmoid'))  # sigmoid评分

#输出网络结构
model.summary()

# 编译模型
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

training_epoches=600

#训练时
steps_per_epoch = train_data_gen.n // train_data_gen.batch_size  # 计算每个epoch要计算的图片个数
history = model.fit(train_data_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=training_epoches,
                    verbose=1, shuffle=True)

#显示损失率以及准确率曲线
plt.plot(history.history['accuracy'])      # 准确率曲线
plt.figure()
plt.plot(history.history['loss'])   # 损失下降曲线
plt.show()

#评价测试集
steps = valid_data_gen.n // valid_data_gen.batch_size
model.evaluate(valid_data_gen, steps=steps)



