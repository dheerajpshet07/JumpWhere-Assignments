# 17. Write a Python program to match key values in two dictionaries.
# Sample dictionary: {'key1': 1, 'key2': 3, 'key3': 2}, {'key1': 1, 'key2': 2}
# Expected output: key1: 1 is present in both x and y


def findMatch(my_dict1,my_dict2):
    for key,val in my_dict1.items():
        for k,v in my_dict2.items():
            if key == k:
                if val == v:
                    print(key," , ",val," is present in both the dictionary")

my_dict1 = {'a': 100, 'b': 200, 'c': 300,'d':100,'e':300}
my_dict2 = {'a': 100, 'b': 200, 'e': 300,'d':100,'f':300}
findMatch(my_dict1,my_dict2)

