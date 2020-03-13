#《Python深度学习：基于tensorflow》—— numpy部分

import numpy as np          #此处引用库
list1 = [3.14,2.17,0,1,2]
nd1=np.array(list1)         #使用numpy中的array
print(nd1)
print(type(nd1))

'''输出 
[3.14 2.17 0.   1.   2.  ]
<class 'numpy.ndarray'>
'''