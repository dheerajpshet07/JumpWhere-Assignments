# 10. Write a Python program to remove a key from a dictionary. 

dict = {'a': 100, 'b': 200, 'c': 300}
key = input("Enter the key to remove : ")
print('Before deleting',dict)
try:
    del dict[key]
    print('Before deleting',dict)
except:
    print("No key exist")

