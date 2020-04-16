# coding=utf-8
#如果下面代码报错可以通过上面的注释块解决
#如果用python3.6其以上无法正常通过，则请使用Python2.7版本，因为Libpython的适配问题
#我的python管理是通过conda搭建多个环境
import theano
from theano import tensor as T
#符号计算图模型
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
'''
variable节点：即符号的变量节点，符号变量是符号表达式存放信息的数据结构，可以分为输入符号和输出符号
type节点：当定义了一种具体的变量类型以及变量的数据类型时，Theano为其指定数据存储的限制条件
apply节点：把某一种类型的符号操作符应用到具体的符号变量中，与variable不同，apply节点无须由用户指定，一个apply节点包括3个字段：op、inputs、outputs
op节点：即操作符节点，定义了一种符号变量间的运算，如+、-、sum()、tanh()等
theano的这些计算图是由Apply和Variable将节点连接而组成，它们分别与函数的应用和数据相连接。操作由op实例表示，而数据类型由type实例表示
'''
