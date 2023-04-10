# 8) Write a python class which has 2 methods get_string and print_string. get_string takes a string from the user and print_string prints the string in reverse order 

class Solution:
    def __init__(self):
        self.input_str = ""

    def getString(self):
        self.input_str = input("Enter a string : ")

    def printString(self):
        print("String in reverse order  ",self.input_str[::-1])

obj = Solution()
obj.getString()
obj.printString()