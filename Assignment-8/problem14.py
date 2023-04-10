# 4) Write a Python program to find if a given string starts with a given character using Lambda.

str = input("Enter a String:\n")
ch = input("ENter a character:\n");

res = lambda str, ch:"Yes" if str.startswith(ch) else "No"
print("Whether the given string ",str," starts with ",ch," : ",res(str,ch))