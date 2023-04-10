# 6)  Write a Python program to find numbers divisible by nineteen or thirteen from a list of numbers using Lambda 
# Orginal list: [19, 65, 57, 39, 152, 639, 121, 44, 90, 190] 
# Numbers of the above list divisible by nineteen or thirteen: [19, 65, 57, 39, 152, 190] 

numList = [19, 65, 57, 39, 152, 639, 121, 44, 90, 190] 
check_divisibility = list(filter(lambda num : (num%13 == 0 or num%19 == 0), numList))
print("The numbers divisible by 13 or 19 is : \n",check_divisibility)