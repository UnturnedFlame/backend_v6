import pickle

dict1 = {'A':[1,2,3],'B':[4,5,6],'C':[7,8,9]}

pickle.dump(dict1, open('dict1.npy', 'wb'))     # 保存

loaded = pickle.load(open('dict1.npy', 'rb'))   # 加载

print(loaded)
print(loaded.get('A'))