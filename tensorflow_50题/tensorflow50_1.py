# coding=utf-8
#导入tensorflow库简写为tf，并输出版本

import tensorflow as tf
tf.__version__

#若遇到报错：Could not load dynamic library 'cudart64_101.dll，则将cuda路径下bin里的cudart64_101改成cudart64_100