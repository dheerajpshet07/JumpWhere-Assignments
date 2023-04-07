# 5. Write a Python program to get a single string from two given strings, separated by a space and swap the first two characters of each string.
# Sample String : 'abc', 'xyz' 
# Expected Result : 'xyc abz'

str1 = input("Enter 1st string: ")
str2 = input("Enter 2nd string: ")

len1 = len(str1)
len2 = len(str2)

finalString = str2[0:len2-1]+str1[len1-1] +" "+ str1[0:len1-1]+ str2[len2-1]
print(finalString)