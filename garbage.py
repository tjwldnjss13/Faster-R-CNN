import numpy as np

a = [0, 1, 2, 3, 4]
for i, j in enumerate(a):
    print(i, j)
    if j == 2:
        a.pop(j)
        i -= 1
