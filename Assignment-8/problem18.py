# 8) Write a Python program to check whether a given string contains a capital letter, a lower case letter, a number and a minimum length using lambda. Minimum length : 10 input string: PaceWisd0m o/p: valid string 

check_string_validity = lambda str : "Valid String" if any(x.isupper() for x in str) and any(x.islower() for x in str) and any(x.isdigit() for x in str) and len(str)>=10 else "Invalid String"

str = input("Enter the string:\n")
print(str, " is ",check_string_validity(str))