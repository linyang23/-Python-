# coding=utf-8
# #tensorflow基础
import tensorflow as tf
##张量
#指定0阶张量的形状
t_list = []
t_tuple = ()
#指定一个长度为2的向量，例如[2,3]
t_1 = [2]
#指定一个2x3矩阵的形状
t_2 = [2, 3]
#表示任意长度的向量
t_3 = [None]
#表示行数任意列数为2的矩阵的形状
t_4 = [None, 2]
#表示第一维长度为3，第二维长度为2，第三维长度任意的三阶张量
t_5 = [3, 2, None]
##计算图
#创建一个新的数据流图
graph = tf.Graph()
#使用with指定其后的一些op添加到这个Graph中
with graph.as_dafault():
    a = tf.add(2, 4)
    b = tf.multiply(2, 4)
#如果有多个Graph，一般采用如下方法
graph1 = tf.Graph()
#使用with指定其后的一些op添加到这个Graph中
with graph1.as_dafault():
    a1 = tf.add(2, 4)
    #定义graph1的op、tensor等
graph2 = tf.Graph()
    # 使用with指定其后的一些op添加到这个Graph中
with graph2.as_dafault():
    a2 = tf.add(2, 4)
    #定义graph2的op、tensor等
 ##对象
 ##创建Session对象
#sess = tf.Session()
#sess.run(fetches, feed_dict = None, run_metadata = None)
a = tf.add(2, 4)
b = tf.multiply(a, 5)
sess = tf.Session()
sess.run(b)
sess.run([a, b])