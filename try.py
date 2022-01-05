import pickle
list = [1,2,3]
list_file = open('list.pickle','wb')
pickle.dump(list,list_file)

