import tensorflow as tf

#计算y=x*x在x=3处的导数
x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)                    #一般在网络中使用时，不需要显式调用watch函数
  y = x * x
dy_dx = g.gradient(y, x)        # y’ = 2*x = 2*3 = 6
print(dy_dx)

