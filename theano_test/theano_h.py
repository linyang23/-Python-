# coding=utf-8
#如果下面代码报错可以通过上面的注释块解决
#如果用python3.6及以上无法正常通过，则请使用Python2.7版本，因为Libpython的适配问题
#我的python管理是通过conda搭建多个环境
#共享变量（共享变量有一个内部状态的值，这个值可以被多个函数共享。它可以存储在显存中，利用GPU提高性能。我们可以使用get_value和set_value方法来读取或者修改共享变量的值，使用共享变量实现累加操作）
import theano
from theano import tensor as T
from theano import shared
import numpy as np
#定义一个共享变量，并初始化为0
state = shared(0)
inc = T.iscalar('inc')
accumulator = theano.function([inc], state, updates = [(state, state + inc)])
#打印state的初始值
print(state.get_value())
accumulator(1)      #进行一次函数调用
#函数返回后，state的值发生了变化
print(state.get_value())
'''
输出：
0
1
'''