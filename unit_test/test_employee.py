import unittest
# from unittest.mock import patch
from unittest import *
from employee import Employee  # Import Employee class from the employee module.


class TestEmployee(unittest.TestCase):
    """
    setUp and tearDown for DRY. Do not repeat yourself. setUp and tearDown are camel cased.
    setUp method will run itself before every single code. The tearDown method will run itself after every single test.
    """

    @classmethod
    def setUpClass(cls):
        print('setupClass')

    @classmethod
    def tearDownClass(cls):
        print('teardownClass')

    def setUp(self):
        """
        Set the instances as instance attributes by putting self.<instance_name>. These will be created before every
        single test.
        """
        print('setUp')
        self.emp_1 = Employee('Corey', 'Schafer', 50000)
        self.emp_2 = Employee('Sue', 'Smith', 60000)

    def tearDown(self):
        """
        Use it in case the test case adds files in a directory or a database. In such cases, in the setUp method we ca
        create those directories and in the tearDown method we can delete those.
        """
        print('tearDown\n')

    def test_email(self):
        print('test_email')
        self.assertEqual(self.emp_1.email, 'Corey.Schafer@email.com')
        self.assertEqual(self.emp_2.email, 'Sue.Smith@email.com')

        self.emp_1.first = 'John'
        self.emp_2.first = 'Jane'

        self.assertEqual(self.emp_1.email, 'John.Schafer@email.com')
        self.assertEqual(self.emp_2.email, 'Jane.Smith@email.com')

    def test_fullname(self):
        print('test_fullname')
        self.assertEqual(self.emp_1.fullname, 'Corey Schafer')
        self.assertEqual(self.emp_2.fullname, 'Sue Smith')

        self.emp_1.first = 'John'
        self.emp_2.first = 'Jane'

        self.assertEqual(self.emp_1.fullname, 'John Schafer')
        self.assertEqual(self.emp_2.fullname, 'Jane Smith')

    def test_apply_raise(self):
        print('test_apply_raise')
        self.emp_1.apply_raise()
        self.emp_2.apply_raise()

        self.assertEqual(self.emp_1.pay, 52500)
        self.assertEqual(self.emp_2.pay, 63000)

    # def test_monthly_schedule(self):
    #     with patch('employee.requests.get') as mocked_get:
    #         mocked_get.return_value.ok = True
    #         mocked_get.return_value.text = 'Success'
    #
    #         schedule = self.emp_1.monthly_schedule('May')
    #         mocked_get.assert_called_with('http://company.com/Schafer/May')
    #         self.assertEqual(schedule, 'Success')
    #
    #         mocked_get.return_value.ok = False
    #
    #         schedule = self.emp_2.monthly_schedule('June')
    #         mocked_get.assert_called_with('http://company.com/Smith/June')
    #         self.assertEqual(schedule, 'Bad Response!')


if __name__ == '__main__':
    unittest.main()

# Tests do not run in an order. That's why we need to keep our tests isolated from one another.

# Best practices
# Sometimes it is also useful to have some code run at the very beginning of the test file. Then have some clean up code
# that runs after all the tests have been run. So, unlike the setUp and tearDown that runs before and after every single
# test it'll be nice if we've something that runs once before anything and then once after everything. We can do this
# with two class methods, called setUp class and tearDownClass.

# Class methods @classmethod
# This means that we are working with the class rather than the instance of the class.
# Note the naming convention of the setUpClass and tearDownClass.
# Use case: We may want to set up a database in the setUp method and tearDown once we are done with the test.

# Best Practices 36:55
# 1. Tests should be isolated: This means that the tests should rely on other tests or affect other tests.
# So we should be able to run a test independent of the other tests, all by itself.
# In this video, the instructor was adding tests to the existing code! In TDD, we write the test before we write the
# code. Think about what you want your code to do. Write the tests implementing that behaviour. Then watch the tests
# fail Write the code in such a way that the code passes all the tests.
