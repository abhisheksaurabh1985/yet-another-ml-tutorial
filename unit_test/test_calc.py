import unittest
import calc

# The test class inherits from unittest.TestCase. It will give us a lot of capabilities for testing which is present
# within that class.


class TestCalc(unittest.TestCase):

    def test_add(self):
        # Method has to start with 'test_'. This is required so that when we run this the application knows which method
        # represents test. Without this, the method won't run. There would be any error but no test would run. I'd
        # recommend you to try this out.
        self.assertEqual(calc.add(10, 5), 15)
        self.assertEqual(calc.add(-1, 1), 0)
        self.assertEqual(calc.add(-1, -1), -2)

    def test_subtract(self):
        self.assertEqual(calc.subtract(10, 5), 5)
        self.assertEqual(calc.subtract(-1, 1), -2)
        self.assertEqual(calc.subtract(-1, -1), 0)

    def test_multiply(self):
        self.assertEqual(calc.multiply(10, 5), 50)
        self.assertEqual(calc.multiply(-1, 1), -1)
        self.assertEqual(calc.multiply(-1, -1), 1)

    def test_divide(self):
        self.assertEqual(calc.divide(10, 5), 2)
        self.assertEqual(calc.divide(-1, 1), -1)
        self.assertEqual(calc.divide(-1, -1), 1)
        self.assertEqual(calc.divide(5, 2), 2.5)

        # The reason we are doing this way is the function will throw the ValueError and our test will think that the
        # function threw an error.
        self.assertRaises(ValueError, calc.divide, 10, 0)
        # the following approach using the context manager allows us to use the function the way we are used to call it.
        with self.assertRaises(ValueError):
            calc.divide(10, 0)


if __name__ == '__main__':
    unittest.main()


# How to run in the terminal?
# python test_calc.py wouldn't work. Paste the output in the blog. Instead, python -m unittest test_calc.py.
# Note the dot in the terminal output. If the test fails, we'll get an F instead of a dot.
# Importance of __main__.
# Testing edge cases. Say sum of a -ve and a +ve number. Sum of a -ve and a +ve number.
# More than the number of test cases, focus should be on writing good test cases.
