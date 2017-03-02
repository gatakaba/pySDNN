from pysdnn.coding import PatternCoding

pc = PatternCoding(binary_vector_dim=100, division_num=10, reversal_num=1)

print(pc._make_binary_vector_table().shape == (10, 100))

print(dir(pc))
