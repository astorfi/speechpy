import scipy.io.wavfile as wav
import numpy as np
import os
import sys
lib_path = os.path.abspath(os.path.join('../..'))
print(lib_path)
sys.path.append(lib_path)
import speechpy
import os
import pytest

@pytest.fixture(scope='session', autouse=True)
def package():
    return speechpy