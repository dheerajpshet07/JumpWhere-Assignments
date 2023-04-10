# 9) Write a Python class named Circle constructed by a radius and two methods which will compute the area and the perimeter of a circle. 

class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius ** 2

    def perimeter(self):
        return 2 * 3.14159 * self.radius


radius = float(input("Enter the value of radius\n"))
my_circle = Circle(radius)
print("Area of the circle:", my_circle.area())
print("Perimeter of the circle:", my_circle.perimeter())

