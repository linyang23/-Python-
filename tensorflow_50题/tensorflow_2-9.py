#2.创建一个3x3的0常量张量
import tensorflow as tf
c = tf.zeros([3, 3])
print(c)

#3.根据上题张量的形状，创建一个一样形状的1常量张量
c_1=tf.ones_like(c)
print(c_1)

#4.创建一个2x3，数值全为6的常量张量
c_2=tf.fill([2, 3], 6)
print(c_2)

#5.创建3x3随机的随机数组
c_3=tf.random.normal([3,3])
print(c_3)

#6.通过二维数组创建一个常量张量
c_4=tf.constant([[1, 2], [3, 4]])  # 形状为 (2, 2) 的二维常量
print(c_4)

#7.取出张量中的numpy数组
c_5=c_4.numpy()
print(c_5)

#8.从1.0-10.0等间距取出5个数形成一个常量张量
c_6=tf.linspace(1.0, 10.0, 5)
print(c_6)

#9.从1开始间隔2取1个数字，到大等于10为止
c_7=tf.range(start=1, limit=10, delta=2)
print(c_7)

'''输出
tf.Tensor(
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]], shape=(3, 3), dtype=float32)
 
tf.Tensor(
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]], shape=(3, 3), dtype=float32)
 
tf.Tensor(
[[6 6 6]
 [6 6 6]], shape=(2, 3), dtype=int32)
 
tf.Tensor(
[[-1.6534834  -0.06619667 -1.180152  ]
 [-0.26551527 -1.130955    1.4781814 ]
 [-0.34386456  0.28343552  2.174725  ]], shape=(3, 3), dtype=float32)
 
tf.Tensor(
[[1 2]
 [3 4]], shape=(2, 2), dtype=int32)
[[1 2]
 [3 4]]
 
tf.Tensor([ 1.    3.25  5.5   7.75 10.  ], shape=(5,), dtype=float32)

tf.Tensor([1 3 5 7 9], shape=(5,), dtype=int32)
'''
