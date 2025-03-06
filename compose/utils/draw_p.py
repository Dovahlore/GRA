import matplotlib.pyplot as plt
import numpy as np
import math
pmax=0
xs=[xs for xs in range(1,30)]

y=[]
for x in xs:
    y.append(pmax-10*np.log10(x)-min(10,10*np.log10(x)))
print(xs,y)
plt.plot(xs,y)
plt.xlabel('N')
plt.ylabel('P')
plt.show()