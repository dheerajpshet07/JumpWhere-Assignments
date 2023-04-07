# 2. Write a Python program to count the number of characters (character frequency) in a string. 
# Sample String : google.com'
# Expected Result : {'o': 3, 'g': 2, '.': 1, 'e': 1, 'l': 1, 'm': 1, 'c': 1}

str = input('Enter the string')
counter = {}
for s in str:
    if s in counter:
        counter[s]+=1
    else:
        counter[s]=1

print(counter)
