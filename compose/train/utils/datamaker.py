import torch
import numpy as np
import pickle
from sympy.codegen.cnodes import sizeof
array=[[],[]]
k=7
for i in range(k):
    for j in range(k):
        if i-1>=0:#上
            id=i*k+j
            nei=(i-1)*k+j
            array[0].append(id)
            array[1].append(nei)
        if i+1<k:#下
            id=i*k+j
            nei=(i+1)*k+j
            array[0].append(id)
            array[1].append(nei)
        if j-1>=0:#左
            id=i*k+j
            nei=i*k+j-1
            array[0].append(id)
            array[1].append(nei)
        if j+1<k:#右
            id=i*k+j
            nei=i*k+j+1
            array[0].append(id)
            array[1].append(nei)

adj=torch.tensor(array)

pickle.dump(adj,open('../edge_index.pkl','wb'))



