#数据合并与展平

#合并一维数组
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.append(a, b)
print(c)
#上面使用append，下面使用concatenate
d = np.concatenate([a, b])
print(d)

#多维数组的合并
import numpy as np
a = np.arange(4).reshape(2, 2)
b = np.arange(4).reshape(2, 2)
#按行合并
c = np.append(a, b, axis = 0)
print(c)
print("合并后数据维度", c.shape)
#按列合并
d = np.append(a, b, axis = 1)
print("按列合并结果：")
print(d)
print("合并后数据维度", d.shape)

#矩阵展平
import numpy as np
nd15 = np.arange(6).reshape(2, -1)
print(nd15)
#按列优先，展平
print("按列优先，展平")
print(nd15.ravel('F'))
#按行优先，展平
print("按行优先，展平")
print(nd15.ravel())

'''
输出：
[1 2 3 4 5 6]
[1 2 3 4 5 6]
[[0 1]
 [2 3]
 [0 1]
 [2 3]]
合并后数据维度 (4, 2)
按列合并结果：
[[0 1 0 1]
 [2 3 2 3]]
合并后数据维度 (2, 4)
[[0 1 2]
 [3 4 5]]
按列优先，展平
[0 3 1 4 2 5]
按行优先，展平
[0 1 2 3 4 5]
'''