# coding=utf-8
import tensorflow as tf

#计算y=x*x在x=3处的导数
x = tf.constant(3.0)            #15.新建常量，值为3.0
with tf.GradientTape() as g:    #16.新建一个GradientTape追踪梯度，把要微分的公式写在里面
  g.watch(x)                    #一般在网络中使用时，不需要显式调用watch函数
  y = x * x
dy_dx = g.gradient(y, x)        #17.求y对于x的导数y’=2*x=2*3=6
print(dy_dx)

