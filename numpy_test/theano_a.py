'''
Theano用符号变量TensorⅤariable来表示变量，又称为张量（Tensor）
张量是Theano的核心元素（也是TensorFlow的核心元素），是Theano表达式和运算操作的基本单位
张量可以是标量（scalar）、向量（vector）、矩阵（matrix）等的统称
具体来说，标量就是我们通常看到的0阶的张量，如12, a等，而向量和矩阵分别为1阶张量和2阶的张量
'''

import theano
'''from theano import tensor as T
#初始化张量
x = T.scalar(name = 'input', dtype = 'float32')
w = T.scalar(name = 'weight', dtype = 'float32')
b = T.scalar(name = 'bias', dtype = 'float32')
z = w * x + b
#编译程序
net_input = theano.function(inputs = [w, x, b], outputs = z)
#执行程序
print('net_input: %2f'% net_input(2.0, 3.0, 0.5))
'''