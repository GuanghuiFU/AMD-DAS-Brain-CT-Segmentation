import numpy as np

read_dictionary = np.load(r"...\img_sim_hash\mix\BRATS_001_79.jpg_mix.npy",allow_pickle=True).item()
print(read_dictionary)

#read the most similar
# a=min(read_dictionary, key=read_dictionary.get)
# print(read_dictionary[a])
# print(a)
#read the Top n similar
n=8
L = sorted(read_dictionary.items() ,key=lambda item :item[1] ,reverse=False)
L = L[:n]
print(L)
dictdata = {}
for l in L:
    dictdata[l[0]] = l[1]
print(dictdata.keys())
