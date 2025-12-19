import unittest
import os
import sys

def run_tests():
    start_dir = 'tests_drone'
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == '__main__':
    run_tests()
