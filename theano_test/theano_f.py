# coding=utf-8
# 如果下面代码报错可以通过上面的注释块解决
# 如果用python3.6其以上无法正常通过，则请使用Python2.7版本，因为Libpython的适配问题
# 我的python管理是通过conda搭建多个环境

#条件判断，Theano是一种符号语言，条件判断不能直接使用Python的if语句
from theano import tensor as T
from theano.ifelse import ifelse
import theano,time,numpy
a, b = T.scalars('a', 'b')
x, y = T.matrices('x', 'y')
z_switch = T.switch(T.lt(a, b), T.mean(x), T.mean(y))       #lt:a<b?
z_lazy = ifelse(T.lt(a, b), T.mean(x), T.mean(y))
#optimizer:optimizer的类型结构（可以简化计算，增加计算的稳定性）
#linker：决定使用哪种方式进行编译（c/python）
f_switch = theano.function([a, b, x, y], z_switch, mode = theano.Mode(linker = 'vm'))
f_lazyifelse = theano.function([a, b, x, y], z_lazy, mode = theano.Mode(linker = 'vm'))
va11 = 0.
va12 = 1.
big_mat1 = numpy.ones((1000, 100))
big_mat2 = numpy.ones((1000, 100))
n_times = 10
tic = time.clock()
for i in range(n_times):
    f_switch(va11, va12, big_mat1, big_mat2)
print('time spent evaluating both values %f sec' % (time.clock() - tic))
tic = time.clock()
for i in range(n_times):
    f_lazyifelse(va11, va12, big_mat1, big_mat2)
print('time spent evaluating one values %f sec' % (time.clock() - tic))
'''
输出：
time spent evaluating both values 0.003085 sec
time spent evaluating one values 0.003461 sec
'''