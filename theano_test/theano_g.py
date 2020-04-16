# coding=utf-8
# 如果下面代码报错可以通过上面的注释块解决
# 如果用python3.6其以上无法正常通过，则请使用Python2.7版本，因为Libpython的适配问题
# 我的python管理是通过conda搭建多个环境

#循环语句（scan是个灵活复杂的函数，任何用循环、递归或者跟序列有关的计算，都可以用scan完成）
import theano
from theano import tensor as T
import numpy as np
#定义单步的函数，实现a*x^n
#输出如参数的顺序要与下面scan的输入参数对应
def one_step(coef, power, x):
    return coef * x ** power
coefs = T.ivector()         #每步变化的值，系数组成的向量
powers = T.ivector()        #每步变化的值，指数组成的向量
x = T.iscalar()     #每步不变的值，自变量
#seq,out_info,non_seq与one_step函数的参数顺序一一对应
#返回的result时每一项的符号表达式组成的list
result, updates = theano.scan(fn = one_step, sequences = [coefs, powers], outputs_info = None, non_sequences = x)
#每一项的值与输入的函数关系
f_poly = theano.function([x, coefs, powers], result, allow_input_downcast = True)
coef_val = np.array([2, 3, 4, 6, 5])
power_val = np.array([0, 1, 2, 3, 4])
x_val = 10
print("多项式各项的值：", f_poly(x_val, coef_val, power_val))
#scan返回的result时每一项的值，并没有求和，如果我们只想要多项式的值，可以把f_poly写成这样：
#多项式每一项的和与输入的函数关系
f_poly = theano.function([x, coefs, powers], result.sum(), allow_input_downcast = True)
print("多项式和的值：", f_poly(x_val, coef_val, power_val))
'''
输出：
('\xe5\xa4\x9a\xe9\xa1\xb9\xe5\xbc\x8f\xe5\x90\x84\xe9\xa1\xb9\xe7\x9a\x84\xe5\x80\xbc\xef\xbc\x9a', array([    2,    30,   400,  6000, 50000]))
('\xe5\xa4\x9a\xe9\xa1\xb9\xe5\xbc\x8f\xe5\x92\x8c\xe7\x9a\x84\xe5\x80\xbc\xef\xbc\x9a', array(56432, dtype=int64))
'''