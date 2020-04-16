# coding=utf-8
#如果下面代码报错可以通过上面的注释块解决
#如果用python3.6其以上无法正常通过，则请使用Python2.7版本，因为Libpython的适配问题
#我的python管理是通过conda搭建多个环境
import theano
from theano import tensor as T
#函数定义的格式
x, y = T.fscalars('x', 'y')
z1 = x + y
z2 = x * y
#定义x、y为自变量，z1、z2为函数返回值
f = theano.function([x, y], [z1, z2])
#返回当x=2，y=3的时候，z1，z2的值
print(f(2, 3))
'''
输出：[array(5., dtype=float32), array(6., dtype=float32)]
'''
#自动求导
x = T.fscalar('x')      #定义一个float变量x
y = 1 / (1 + T.exp(-x))
dx = theano.grad(y, x)      #偏导数函数
f = theano.function([x], dx)        #定义函数f，输入为x，输出为y的偏导数
print(f(3))     #计算当x=3时，函数y的偏导数
'''
输出：0.04517666
'''
#更新共享变量参数
w = theano.shared(1)    #定义一个共享变量w，其初始值为1
x = T.iscalar('x')
f = theano.function([x], w, updates = [[w, w + x]])     #定义函数自白能量为x，因变量为y，当函数执行完毕后，更新参数w=w+x
print(f(3))     #输出更新前的w
print(w.get_value())        #输出更新后的w
'''
输出：
1
4
'''
