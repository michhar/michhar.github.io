"""
testing try statements
"""

print('------------- Using a try/except/else -------------')


# Simple try/except and else statements
try:
    assert 1 == 2
except AssertionError:
    print('Received an AssertionError.  Try again.  No else will run.')
else:
    # I will only run if the try block completes without an exception
    print('All is well in the try block.')

print('\n------------- Using a try/except/finally -------------')

try:
    assert 1 == 2
except AssertionError:
    print('Received an AssertionError.  The finally will still run...')
finally:
    # I will always run
    print('This finally is always run regardless of exceptions in the try.')


# # For loops with else
# for a in range(10):
#     if a < 5:
#         print('a is less than 5, now break.  No else will run.')
#         break
# else:
#     print('The for loop did not break.')

