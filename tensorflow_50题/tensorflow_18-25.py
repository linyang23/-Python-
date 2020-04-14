import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#总任务生成添加随机噪声的沿100个[y=3x+2]的数据点，再对这些数据点进行拟合

#生成数据
x=tf.random.normal([100,1]).numpy()
noise=tf.random.normal([100,1]).numpy()
y=3*x+2+noise

#可视化
plt.scatter(x,y)

#创建需要预测的参数w，b
w=tf.Variable(np.random.randn())
b=tf.Variable(np.random.randn())
print('w:%f,b:%f'%(w.numpy(),b.numpy()))

#创建线性回归预测模型
def linear_regression(x):
    return w*x+b

#创建损失函数，此处采用真实值与预测值的差的平方和
def mean_square(y_pred,y_true):
    return tf.reduce_mean(tf.square(y_pred-y_true))

#创建GradientTape，写入需要微分的过程
with tf.GradientTape() as tape:
    pred=linear_regression(x)
    loss=mean_square(pred,y)

#对loss，分别求关于w，b的偏导数
dw,db=tape.gradient(loss,[w,b])

#用最简单朴素的梯度下降更新w，吧，learning_rate设置为0.1
w.assign_sub(0.1*dw)
b.assign_sub(0.1*db)
print('w:%f,b:%f'%(w.numpy(),b.numpy()))

#将以上的单次迭代过程连续循环迭代20次，并记录每次的loss，w，b
for i in range(20):
    with tf.GradientTape() as tape:
        pred = linear_regression(x)
        loss = mean_square(pred, y)
    dw, db = tape.gradient(loss, [w, b])
    w.assign_sub(0.1 * dw)
    b.assign_sub(0.1 * db)
    print("step: %i, loss: %f, W: %f, b: %f" % (i + 1, loss, w.numpy(), b.numpy()))

#画出最终拟合的曲线
plt.plot(x,y,'ro',label='Original data')
plt.plot(x,np.array(w*x+b),label='Fltted line')
plt.legend()
plt.show()