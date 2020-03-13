import tensorflow as tf
a = tf.constant([[1, 2], [3, 4]])

#10.将两个张量相加
c_1=a + a

#11.将两个张量做矩阵乘法
c_2=tf.matmul(a, a)

#12.两个张量做点乘
c_3=tf.multiply(a, a)

#13.将一个张量转置
c_4=tf.linalg.matrix_transpose(a)

#14_1.将一个12x1张量变形成3行的张量
b = tf.linspace(1.0, 10.0, 12)
c_5=tf.reshape(b,[3,4])

#14_2.方法二
c_6=tf.reshape(b,[3,-1])

#显示
print(c_1,c_2,c_3,c_4,c_5,c_6)

'''输出
tf.Tensor(
[[2 4]
 [6 8]], shape=(2, 2), dtype=int32) 
 tf.Tensor(
[[ 7 10]
 [15 22]], shape=(2, 2), dtype=int32) 
 tf.Tensor(
[[ 1  4]
 [ 9 16]], shape=(2, 2), dtype=int32) 
 tf.Tensor(
[[1 3]
 [2 4]], shape=(2, 2), dtype=int32) 
 tf.Tensor(
[[ 1.         1.8181818  2.6363635  3.4545455]
 [ 4.272727   5.090909   5.909091   6.7272725]
 [ 7.5454545  8.363636   9.181818  10.       ]], shape=(3, 4), dtype=float32) 
 tf.Tensor(
[[ 1.         1.8181818  2.6363635  3.4545455]
 [ 4.272727   5.090909   5.909091   6.7272725]
 [ 7.5454545  8.363636   9.181818  10.       ]], shape=(3, 4), dtype=float32)
'''