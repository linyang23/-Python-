#文件存取
import numpy as np
nd9=np.random.random([5,5])
np.savetxt(X=nd9,fname='./test2.txt')       #将数据保存到txt中，并命名为test2.txt
nd10=np.loadtxt('./test2.txt')              #从txt中读取数据
print(nd10)