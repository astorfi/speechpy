import os
import sys
lib_path = os.path.abspath(os.path.join('../..'))
print(lib_path)
sys.path.append(lib_path)



# content of test_class.py
class TestClass(object):
    def test_one(self):
        x = "this"
        assert 'h' in x

    def test_two(self):
        x = "hello"
        assert hasattr(x, 'check')
