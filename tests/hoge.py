from pysdnn.coding import PatternCoding
import numpy as np

pc = PatternCoding(binary_vector_dim=100, division_num=10, reversal_num=1)
pc.make_binary_vector_table(3)
print(pc.binary_vector_table.shape)

print(pc.num_to_pattern(np.array([0.5, 0.2, 0.1])).shape)
