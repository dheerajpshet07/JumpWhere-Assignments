# 4. Write a Python script to check if a given key already exists in a dictionary. 

my_dict = { 'num6': 6, 'num3': 3, 'num2': 2, 'num4': 4, 'num1': 1, 'num5': 5}

dict_key = input("Enter the key : ")
retVal =  my_dict.get(dict_key)

if retVal is not None:
    print("Key Exists")
else:
    print("Key Does not Exists")
