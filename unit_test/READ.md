---
layout: post
title: Best practices for unit testing in Python
tags: [Python, Unit Test, Software Engineering]
---
<p align="justify">
In this post we will learn how to write unit tests in Python and also a few best practices. I had initially thought to
write this post only on the best practices. But later thought to start with the very basics. So, if you are someone who
has been using the <code>print</code> statements to debug your code, then you've come to the right place. I will start with the hello-world-ish example of unit testing i.e. an example of a tutorial. Then while discussing the best practices, we'll take up a relatively complex example. However, because of covering the basics as well, this post has become longer than my usual posts. You can find the code explained here in __[this]()__ Github repository.
</p>

<p align="justify">
Note that testing the code isn't the most exciting thing to do. Nevertheless, it helps us save a lot of time going forward.
Also note that more than the sheer quantity of test cases, it's the quality of the test cases which gives you a confidence that your updates and refactoring doesn't have any unintended consequences or break your code in any way. Good test cases helps us quickly see if code is working the way it should.
</p>

<p align="justify">
In this post we'll see how the unit testing works with the built-in <code>unittest</code> module in Python. Without further ado, let's get started. We have a file <code>calc.py</code>  which has a few simple function to add, subtract, multiply and divide. This simple example will help us focus on unit testing.
</p>

<p align="justify">
To start with unit testing, we'll create a file <code>test_calc.py</code> in the current directory. Note that it's the naming convention to start the test file name with <code>test_</code> followed by what you are testing (which in our example is <code>calc</code>). After importing the <code>unittest</code> module and the module which has to be tested, we will start writing our test cases.
</p>

<p align="justify">
In order to create the test cases, we'll first create a test class <code>TestCalc</code> which inherits from the <code>unittest</code> module. Inside the class, we will write the <code>test</code> methods. Similar to the test modules name, the method name should start with <code>test_</code>. This naming convention is required so that when we run this, the system knows which method represents tests. Without this naming convention, the test method won't run. The compiler won't throw any error but no test would run. I'd recommend you to test it out yourself by ignoring the convention once.
</p>

<p align="justify">
Within the test methods, we can use several <code>assert</code> methods defined in the <code>TestCase</code> class for checking and reporting failures. The <code>assert</code> methods have a self-explanatory name. The various <code>assert</code> methods used to test the four functions defined in the <code>test_calc.py</code> file are shown in the snippet below.
</p>

```python
class TestCalc(unittest.TestCase):

    def test_add(self):
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

        # self.assertRaises(ValueError, calc.divide, 10, 0)  # See explanation on this in the paragraph below.
        with self.assertRaises(ValueError):
            calc.divide(10, 0)
```

Note the method to test <code>divide by zero</code> case uses the assertion differently than the other cases. In the <code>calc.py</code> file we have defined that the function should raise a <code>ValueError</code> in such a case. We can test it in either of the following two ways. One, by passing the error as an argument to the assertion method as shown below.
```python
self.assertRaises(ValueError, calc.divide, 10, 0)
```
Note that we are not passing 10 and 0 as an argument to the <code>calc.divide</code> method. Instead, we are passing the argument to the <code>assertRaises</code> method.

There is another way of defining such a test case which I find more intuitive which uses a context manager. It allows us
to call the function the way we are used to call it.
```python
        with self.assertRaises(ValueError):
            calc.divide(10, 0)
```
