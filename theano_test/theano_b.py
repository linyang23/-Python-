# coding=utf-8
#如果下面代码报错可以通过上面的注释块解决
#如果用python3.6其以上无法正常通过，则请使用Python2.7版本，因为Libpython的适配问题
#我的python管理是通过conda搭建多个环境
import numpy as np
import theano
from theano import tensor as T
#使用内置变量类型创建(内置的变量类型只能处理4维及以下的变量)
x = T.scalar(name = 'input', dtype = 'float32')
data = T.vector(name = 'data', dtype = 'float64')
#自定义变量类型
mytype = T.TensorType('float64', broadcastable = (), name = None, sparse_grad = False)
#Python类型变量或者NumPy类型变量转化为Theano共享变量
data1 = np.array([[1, 2], [3, 4]])
shared_data = theano.shared(data1)
type(shared_data)