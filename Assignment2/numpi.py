# import numpy as np
# from collections import Counter

# # Create your matrix
# a = np.array([[ 1,  0, -1],
#               [ 1,  1,  0],
#               [-1,  0,  1],
#               [ 0,  1,  0]])

# # Loop on each column to get the most frequent element and its count
# for i in range(a.shape[1]):
#     count = Counter(a[:, i])
#     count.most_common(1)

import math
import sysconfig
from collections import Counter
import numpy as np 

emission_dict = {'S': {'Y': 2, 'Z': 1}, 'C': {'X': 1, 'Y': 2, 'Z': 1}, 'A': {'X': 2, 'Y': 1, 'Z': 2}, 'T': {'X': 2, 'Z': 2}, 'G': {'X': 1, 'Y': 3}}

trans = ['<s>', 'A', 'C', 'G', 'S', 'T']
print(trans)
trans[1], trans[-1] = trans[-1], trans[1]

print(trans)