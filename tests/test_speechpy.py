import os
import sys


def test_fm_methods_exist(package):
    module = package.feature
    assert hasattr(module, "_feature_most_alternative")
