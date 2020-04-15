#通过random.choice从指定的样本中进行随机抽取数据
import numpy as np
from numpy import random as nr
a = np.arange(1, 25, dtype = float)
c1 = nr.choice(a, size = (3, 4))        #size指定输出数组形状
c2 = nr.choice(a, size = (3, 4), replace = False)       #replace缺省时为True，即可重复抽取
c3 = nr.choice(a, size = (3, 4), p = a / np.sum(a))     #参数p制定每个元素的抽取概率，默认每个元素相同
print("随机可重复抽取")
print(c1)
print("随机但不重复抽取")
print(c2)
print("随机但按制度概率抽取")
print(c3)

'''
输出：
随机可重复抽取
[[14. 13.  6. 18.]
 [15. 24. 13.  3.]
 [19.  6.  2. 11.]]
随机但不重复抽取
[[15. 19.  7. 16.]
 [12.  9.  1.  2.]
 [ 5. 11. 21. 13.]]
随机但按制度概率抽取
[[24. 22. 24. 21.]
 [22. 24. 14. 23.]
 [ 9. 11. 23. 18.]]
'''