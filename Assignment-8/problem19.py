# 9) Write a Python program to find the elements of a given list of strings that contain specific substring using lambda. 
# Original list: ['red', 'black', 'white', 'green', 'orange'] 
# Substring to search: ack Elements of the said list that contain specific substring: ['black'] Substring to search: abc Elements of the said list that contain specific substring: [] 

str_list =  ['red', 'black', 'white', 'green', 'orange'] 
find_substring = lambda str_list, ss:list(filter(lambda str: True if ss in str else False, str_list))

ss = input("Enter a substring\n")
print("List of string that contains ",ss," as a sunstring are:\n",find_substring(str_list,ss))