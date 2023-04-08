# 16. Write a Python program to find the highest 3 values in a dictionary.

def findHighestThree(my_dict):
    num_list = []
    
    for key,val in my_dict.items():
        num_list.append(val)

    num_list.sort(reverse=True)
    print("Highest Three vals are : ",num_list[0]," ",num_list[1]," ",num_list[2])


my_dict = {'a': 100, 'b': 200, 'c': 300,'d':100,'e':300}
findHighestThree(my_dict)