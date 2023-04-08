# 4. Write a Python program to check a triangle is equilateral, isosceles or scalene.
# Note :
# An equilateral triangle is a triangle in which all three sides are equal.
# A scalene triangle is a triangle that has three unequal sides.
# An isosceles triangle is a triangle with two equal sides.
# Expected Output:
# Input lengths of the triangle sides:                                    
# x: 6                                                                    
# y: 8                                                                    
# z: 12                                                                   
# Scalene triangle 
def identifyTriangle(x,y,z):
    if x==y==z:
        return "equilateral"
    elif x==y or x==z or y==z:
        return "isosceles"
    else:
        return "scalene"

x = int(input("Enter the value of 1st side: "))
y = int(input("Enter the value of 2nd side: "))
z = int(input("Enter the value of 3rd side: "))

print(identifyTriangle(x,y,z))