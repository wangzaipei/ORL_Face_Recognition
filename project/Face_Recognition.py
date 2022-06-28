import itertools

import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from tensorflow.python.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report

np.random.seed(2022)  # 随机种子


validation_split = 0.3    # 表示拆分数据30% 做测试集
#定义一个图片生成器，加载以及增强图片
image_generator = ImageDataGenerator(rescale=1. / 255., validation_split=validation_split)
# 图片尺寸
im_height = 112
im_width = 92
#图片路径
image_path='data'
#批处理图片数量
batch_size=50
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
                                                     shuffle=False,
                                                     target_size=(im_height, im_width),
                                                     class_mode='categorical',
                                                     subset='validation')

# pool_size = (2, 2)  # 池化层的大小
# kernel_size = (3, 3)  # 卷积核的大小
input_shape = (im_height, im_width,1)  # 输入图片的维度
# nb_classes = 40  # 分类数目

# 构建模型
model = Sequential()
model.add(Conv2D(24,3,input_shape=input_shape,activation='relu'))  # 卷积层1
model.add(MaxPooling2D(pool_size=2))  # 池化层
model.add(Conv2D(48,3,strides=1,activation='relu'))  # 卷积层2
model.add(MaxPooling2D(pool_size=2))  # 池化层
model.add(Flatten())  # 拉成一维数据
#全连接层
model.add(Dense(2024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(40, activation='softmax'))

#输出网络结构
model.summary()

# 编译模型
model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

training_epoches=30

#训练时
steps_per_epoch = train_data_gen.n // train_data_gen.batch_size  # 计算每个epoch要计算的图片个数
history = model.fit(train_data_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=training_epoches,
                    validation_data=valid_data_gen
                    )

#模型保存
# model.save('model.h5')

#评价测试集
steps = valid_data_gen.n // valid_data_gen.batch_size
model.evaluate(valid_data_gen, steps=steps)

# print(history.history.keys())
#显示损失率以及准确率曲线
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


#模型评估
y_pred=model.predict(valid_data_gen)  #预测分类标签
y_pred = tf.argmax(y_pred, 1) # 独热编码转换为预测标签

report=classification_report(valid_data_gen.classes, y_pred)
Acc=accuracy_score(valid_data_gen.classes, y_pred)
Pre=precision_score(valid_data_gen.classes, y_pred,average='macro')
Rec=recall_score(valid_data_gen.classes, y_pred,average='macro')
cm = confusion_matrix(valid_data_gen.classes, y_pred)
print(report)
print('准确率：',Acc)
print('精确率：',Pre)
print('召回率：',Rec)

#绘制混淆矩阵函数
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        print('')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()

#输出混淆矩阵
print(cm)
#绘画混淆矩阵
plt.figure()
# plt.subplot(2,2,1)
plot_confusion_matrix(cm[1:40,1:40], classes=[0,1,2,3,4,5,6,7,8,9,
                                              10,11,12,13,14,15,16,17,18,19,
                                              20,21,22,23,24,25,26,27,28,29,
                                              30,31,32,33,34,35,36,37,38,39],
                      title='Confusion matrix--Face0-39')
# plt.subplot(2,2,2)
# plot_confusion_matrix(cm[11:20,11:20], classes=[10,11,12,13,14,15,16,17,18,19],
#                       title='Confusion matrix--Face10-19')
# plt.subplot(2,2,3)
# plot_confusion_matrix(cm[21:30,21:30], classes=[20,21,22,23,24,25,26,27,28,29],
#                       title='Confusion matrix--Face20-29')
# plt.subplot(2,2,4)
# plot_confusion_matrix(cm[31:40,31:40], classes=[30,31,32,33,34,35,36,37,38,39],
#                       title='Confusion matrix--Face30-39')

print('预测值：')
print(y_pred)
print('真实值：')
print(valid_data_gen.classes)

plt.show()


