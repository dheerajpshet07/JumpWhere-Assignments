# 6. Write a Python program to add 'ing' at the end of a given string (length should be at least 3). If the given string already ends with 'ing' then add 'ly' instead. If the string length of the given string is less than 3, leave it unchanged.
# Sample String : 'abc'
# Expected Result : 'abcing' 
# Sample String : 'string'
# Expected Result : 'stringly'

str = input("Enter the string")
lenS = len(str)
finalString = str
if(lenS < 3):
    print(str)
else:
    if str[lenS-3]=="i" and str[lenS-2]=="n" and str[lenS-1]=="g":
        finalString += "ly"
        print(finalString)
    else:
        finalString+="ing"
        print(finalString)


