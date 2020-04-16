# coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

#总任务为在CIFAR10训练集上，训练LeNet5模型

#26.定义第1步卷积层的参数
conv1_w = tf.Variable(tf.random.truncated_normal([5, 5, 3, 6],stddev = 0.1))
conv1_b = tf.Variable(tf.zeros([6]))

#27.定义第3步卷积层的层数
conv2_w = tf.Variable(tf.random.truncated_normal([5, 5, 6, 16], stddev = 0.1))
conv2_b = tf.Variable(tf.zeros([16]))

#28.定义第5步全连接层的层数
fc1_w = tf.Variable(tf.random.truncated_normal([5 * 5 * 16, 120], stddev = 0.1))
fc1_b = tf.Variable(tf.zeros([120]))

#29.定义第6步全连接层的参数
fc2_w = tf.Variable(tf.random.truncated_normal([120, 84],stddev = 0.1))
fc2_b = tf.Variable(tf.zeros([84]))

#30.定义第7步全连接层的参数
fc3_w = tf.Variable(tf.random.truncated_normal([84, 10],stddev = 0.1))
fc3_b = tf.Variable(tf.zeros([10]))

def lenet5(input_img):
    #31.搭建INPUT->C1的步骤
    conv1_1 = tf.nn.conv2d(input_img, conv1_w, strides = [1, 1, 1, 1], padding = "VALID")
    conv1_2 = tf.nn.relu(tf.nn.bias_add(conv1_1,conv1_b))

    ##32.搭建C1->S2的步骤
    pool1 = tf.nn.max_pool(conv1_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")

    ##33.搭建S2->C3的步骤
    conv2_1 = tf.nn.conv2d(pool1, conv2_w, strides = [1,1,1,1], padding = "VALID")
    conv2_2 = tf.nn.relu(tf.nn.bias_add(conv2_1,conv2_b))

    ##34.搭建C3->S4的步骤
    pool2 = tf.nn.max_pool(conv2_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")

    ##35.将S4的输出扁平化
    reshaped = tf.reshape(pool2, [-1, 16 * 5 * 5])

    ##35.搭建S4->C5的步骤
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)

    ##36.搭建C5->F6的步骤
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b)

    ##37.搭建F6->OUTPUT的步骤
    OUTPUT = tf.nn.softmax(tf.matmul(fc2, fc3_w) + fc3_b)

    return OUTPUT

#38.创建一个Adam优化器，学习率0.02
optimizer = tf.optimizers.Adam(learning_rate=0.02)

#随机设置测试数据x,y，x的形状为(1，32，32，3)，y的形状为(10)
test_x = tf.Variable(tf.random.truncated_normal([1, 32, 32, 3]))
test_y = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#将数据送入模型，进行反向传播
with tf.GradientTape() as tape:
    #40.将数据传入模型
    prediction = lenet5(test_x)
    print("第一次预测：", prediction)
    #41.使用交叉熵作为损失函数，计算损失
    cross_entropy = tf.reduce_sum(test_y * tf.math.log(prediction))

#42.计算梯度
trainable_variables = [conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b]
grads = tape.gradient(cross_entropy, trainable_variables)

#43.更新梯度
optimizer.apply_gradients(zip(grads, trainable_variables))

print("反向传播后的预测：", lenet5(test_x))

#读入数据，预处理
def load_cifar_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding = 'iso-8859-1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
        Y = np.array(Y)
        return X, Y
def load_cifar(ROOT):
    #data_X, data_Y = load_cifar_batch('/home/kesci/input/cifar10/data_batch_1')     #此路径为在线数据集路径，需在和鲸社区虚拟运行
    data_X, data_Y = load_cifar_batch('D:\学习\数据集\data_batch_1')     #此路径为本地数据集路径，需按照自己数据集位置进行修改
    for b in range(2,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        batch_X, batch_Y = load_cifar_batch(f)
        data_X = np.concatenate([data_X, batch_X])
        data_Y = np.concatenate([data_Y, batch_Y])
    data_test_X, data_test_Y  = load_cifar_batch(os.path.join(ROOT, 'test_batch'))
    return data_X, data_Y, data_test_X, data_test_Y

#train_X, train_Y, test_X, test_Y = load_cifar('/home/kesci/input/cifar10')     #此路径为在线数据集路径，需在和鲸社区虚拟运行
train_X, train_Y, test_X, test_Y = load_cifar('D:\学习\数据集')           #此路径为本地数据集路径，需按照自己数据集位置进行修改
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
train_X.shape, train_X.shape, test_X.shape, test_Y.shape

#捞一个数据看看样子
plt.imshow(train_X[0])
plt.show()
print(classes[train_Y[0]])

#44.预处理1：将train_y, test_y进行归一化
train_X = tf.cast(train_X, dtype=tf.float32) / 255
test_X = tf.cast(test_X, dtype=tf.float32) / 255

#45.预处理2：将train_y, test_y进行onehot编码

train_Y = tf.one_hot(train_Y, depth = 10)
test_Y = tf.one_hot(test_Y, depth = 10)

#因为前面实验的时候修改过参数，所以需要重新初始化所有参数
conv1_w = tf.Variable(tf.random.truncated_normal([5,5,3,6], stddev=0.1))
conv1_b = tf.Variable(tf.zeros([6]))
conv2_w = tf.Variable(tf.random.truncated_normal([5, 5, 6, 16], stddev=0.1))
conv2_b = tf.Variable(tf.zeros([16]))
fc1_w = tf.Variable(tf.random.truncated_normal([5*5*16, 120], stddev=0.1))
fc1_b = tf.Variable(tf.zeros([120]))
fc2_w = tf.Variable(tf.random.truncated_normal([120, 84], stddev=0.1))
fc2_b = tf.Variable(tf.zeros([84]))
fc3_w = tf.Variable(tf.random.truncated_normal([84, 10], stddev=0.1))
fc3_b = tf.Variable(tf.zeros([10]))

#重新定义一个优化器
optimizer2 = tf.optimizers.Adam(learning_rate=0.002)

#写一个算准确率的函数
def accuracy_fn(y_pred, y_true):
    preds = tf.argmax(y_pred, axis=1)  # 取值最大的索引，正好对应字符标签
    labels = tf.argmax(y_true, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

#把数据送入模型，开始训练，训练集迭代5遍，每遍分成25个batch，数据集每迭代完一遍，输出一次训练集上的准确率
EPOCHS = 5  # 整个数据集迭代次数

for epoch in range(EPOCHS):
    for i in range(25):  # 一整个数据集分为10个小batch训练
        with tf.GradientTape() as tape:
            prediction = lenet5(train_X[i * 2000:(i + 1) * 2000])
            cross_entropy = -tf.reduce_sum(train_Y[i * 2000:(i + 1) * 2000] * tf.math.log(prediction))

        trainable_variables = [conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b]  # 需优化参数列表
        grads = tape.gradient(cross_entropy, trainable_variables)  # 计算梯度

        optimizer2.apply_gradients(zip(grads, trainable_variables))  # 更新梯度

    # 每训练完一次，输出一下训练集的准确率
    accuracy = accuracy_fn(lenet5(train_X), train_Y)
    print('Epoch [{}/{}], Train loss: {:.3f}, Test accuracy: {:.3f}'
          .format(epoch + 1, EPOCHS, cross_entropy / 2000, accuracy))

#47.在测试集上进行预测
test_prediction = lenet5(test_X)
test_acc = accuracy_fn(test_prediction, test_Y)
test_acc.numpy()

#取一些数据查看预测结果
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(test_X[i], cmap=plt.cm.binary)
    title = classes[np.argmax(test_Y[i])]+'=>'
    title += classes[np.argmax(test_prediction[i])]
    plt.xlabel(title)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

'''
输出结果：
第一次预测： tf.Tensor(
[[0.24850117 0.09553912 0.11700701 0.02298369 0.15716557 0.08251186
  0.03089823 0.13077648 0.07066613 0.04395071]], shape=(1, 10), dtype=float32)
反向传播后的预测： tf.Tensor(
[[3.4140783e-08 4.0528557e-04 2.9566936e-02 2.1496642e-06 8.2095891e-01
  4.7968952e-03 6.3553260e-04 1.4321260e-01 2.0807283e-04 2.1367386e-04]], shape=(1, 10), dtype=float32)
frog
Epoch [1/5], Train loss: 1.942, Test accuracy: 0.295
Epoch [2/5], Train loss: 1.800, Test accuracy: 0.371
Epoch [3/5], Train loss: 1.670, Test accuracy: 0.412
Epoch [4/5], Train loss: 1.567, Test accuracy: 0.431
Epoch [5/5], Train loss: 1.488, Test accuracy: 0.457
'''