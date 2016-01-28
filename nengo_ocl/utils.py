import numpy as np

from nengo.utils.compat import PY2


def as_ascii(string):
    if not PY2 and isinstance(string, bytes):  # Python 3
        return string.decode('ascii')
    elif not isinstance(string, str):
        return str(string)
    else:
        return string


def indent(s, i):
    return '\n'.join([(' ' * i) + line for line in s.split('\n')])

def round_up2(x,n):
    r = n % x;
    if r == 0:
        return n;
    else:
        return x + n - r;
def round_up(x, n):

    return int(np.ceil(float(x) / n)) * n


def split(iterator, criterion):
    """Returns a list of objects that match criterion and those that do not.
    """
    a = []
    b = []
    for x in iterator:
        if criterion(x):
            a.append(x)
        else:
            b.append(x)

    return a, b
