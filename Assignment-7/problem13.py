# 13. Write a Python program to remove duplicates from Dictionary.
my_dict = {'a': 100, 'b': 200, 'c': 300,'d':100,'e':300}
res = {}
for i in my_dict:
    res.setdefault(my_dict[i],i)

print("After remving duplicates: ", dict((val,key) for key,val in res.items()) )