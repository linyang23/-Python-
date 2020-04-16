# coding=utf-8
import tensorflow as tf

#总任务：实现最简单的保存&读取变量值

fc3_w = tf.Variable(tf.random.truncated_normal([84, 10], stddev=0.1))
fc3_b = tf.Variable(tf.zeros([10]))

#48.新建一个Checkpoint对象，并且往其中灌一个刚刚训练完的数据
save = tf.train.Checkpoint()
save.listed = [fc3_b]
save.mapped = {'fc3_b': save.listed[0]}

#49利用save()的方法保存，并且记录返回的保存路径
save_path = save.save('/home/kesci/work/data/tf_list_example')
print(save_path)


#50.新建一个Checkpoint对象，从里读出数据
restore = tf.train.Checkpoint()
fc3_b2 = tf.Variable(tf.zeros([10]))
print(fc3_b2.numpy())
restore.mapped = {'fc3_b': fc3_b2}
restore.restore(save_path)
print(fc3_b2.numpy())

'''
输出：
/home/kesci/work/data/tf_list_example-1
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
'''