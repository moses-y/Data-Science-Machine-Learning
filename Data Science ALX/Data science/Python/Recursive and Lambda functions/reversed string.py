def r(s):
    if len(s) == 0:
        return s
    return r(s[1:]) + s[0]

# Correct usage
reversed_string = r('programming')  # Reverses the string 'programming'
print(reversed_string)               # Output will be 'gnimmargorP'

class Circle:
    def __init__(self, radius):
        self.radius = radius

    def circumference(self):
        return 2 * 3.14159 * self.radius

# Correct usage
circle = Circle(3)  # Create a circle with radius 3
circumference = circle.circumference()  # Calculate the circumference
print(circumference)  # Output will be approximately 18.85

class Rectangle:

    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

    def area(self):
        return self.width * self.height


rect = Rectangle(3, 4)
print("Area:", rect.area(), "Perimeter:", rect.perimeter())

numbers = [1, 2, 3, 4, 5]
result = map(lambda x: x**2, numbers)
print(list(result))

class ExampleClass:
    def __init__(self,value):
        self.value = value
    def get_value(self):
        return self.value
    

'''The following Python code is meant to calculate the total area of two different shapes, but it contains an error. Identify and correct the error.

'''
class Shape:
    def __init__(self, name):
        self.name = name

class Circle(Shape):
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius

    def area(self):
        return 3.14 * self.radius * self.radius
class Square(Shape):
    def __init__(self, side):
        super().__init__("Square")
        self.side = side

    def area(self):
        return self.side * self.side

circle = Circle(3)
square = Square(4)
total_area = circle.area() + square.area()
print("Total Area:", total_area)
