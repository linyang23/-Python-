'''
Theano�÷��ű���Tensor��ariable����ʾ�������ֳ�Ϊ������Tensor��
������Theano�ĺ���Ԫ�أ�Ҳ��TensorFlow�ĺ���Ԫ�أ�����Theano���ʽ����������Ļ�����λ
���������Ǳ�����scalar����������vector��������matrix���ȵ�ͳ��
������˵��������������ͨ��������0�׵���������12, a�ȣ��������;���ֱ�Ϊ1��������2�׵�����
'''

import theano
'''from theano import tensor as T
#��ʼ������
x = T.scalar(name = 'input', dtype = 'float32')
w = T.scalar(name = 'weight', dtype = 'float32')
b = T.scalar(name = 'bias', dtype = 'float32')
z = w * x + b
#�������
net_input = theano.function(inputs = [w, x, b], outputs = z)
#ִ�г���
print('net_input: %2f'% net_input(2.0, 3.0, 0.5))
'''