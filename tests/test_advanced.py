from context import speechpy

import unittest


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_thoughts(self):
        self.assertIsNone(speechpy.mfcc())


if __name__ == '__main__':
    unittest.main()
