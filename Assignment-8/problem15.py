# 5) Write a Python program to check whether a given string is number or not using Lambda. 

is_number = lambda x:x.replace('.','',1).isdigit() or x.startswith("-") and x[1:].replace(".","",1).isdigit()

print(is_number("56"))
print(is_number("-56"))
print(is_number("5.6"))
print(is_number("mite"))


