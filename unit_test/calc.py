def add(x, y):
    """Add Function"""
    return x + y


def subtract(x, y):
    """Subtract Function"""
    return x - y


def multiply(x, y):
    """Multiply Function"""
    return x * y


def divide(x, y):
    """Divide Function"""
    if y == 0:
        raise ValueError('Can not divide by zero!')
    return x / y

# Create a test module. Create  a new file in the current directory. Naming convention with writing tests is to start
# with 'test_' followed by what you're testing.
# Import calc.py in the test file. Since these files are in the same directory, it can be imported.
# Create a test class that inherits from unittest.testcase.