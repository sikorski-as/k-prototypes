from contextlib import contextmanager

import numpy as np


@contextmanager
def print_array_on_one_line():
    oldoptions = np.get_printoptions()
    np.set_printoptions(linewidth=np.inf)
    yield
    np.set_printoptions(**oldoptions)
