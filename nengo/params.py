import collections
import inspect

import numpy as np

from nengo.utils.compat import (
    is_array, is_integer, is_number, is_string, itervalues)
from nengo.utils.numpy import array_hash, compare
from nengo.utils.stdlib import WeakKeyIDDictionary, checked_call


class DefaultType:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

Default = DefaultType("Default")
ConnectionDefault = DefaultType("ConnectionDefault")
Unconfigurable = DefaultType("Unconfigurable")


def is_param(obj):
    return isinstance(obj, Parameter)


class Parameter(object):
    """Simple descriptor for storing configuration parameters.

    Parameters
    ----------
    default : object
        The value returned if the parameter hasn't been explicitly set.
    optional : bool, optional
        Whether this parameter accepts the value None. By default,
        parameters are not optional (i.e., cannot be set to ``None``).
    readonly : bool, optional
        If true, the parameter can only be set once.
        By default, parameters can be set multiple times.
    """
    equatable = False

    def __init__(self, default=Unconfigurable, optional=False, readonly=None):
        self.default = default
        self.optional = optional

        if readonly is None:
            # freeze Unconfigurables by default
            readonly = default is Unconfigurable
        self.readonly = readonly

        # default values set by config system
        self._defaults = WeakKeyIDDictionary()

        # param values set on objects
        self.data = WeakKeyIDDictionary()

    def __contains__(self, key):
        return key in self.data or key in self._defaults

    def __delete__(self, instance):
        del self.data[instance]

    def __get__(self, instance, type_):
        if instance is None:
            # Return self so default can be inspected
            return self
        if not self.configurable and instance not in self.data:
            raise ValueError("Unconfigurable parameters have no defaults. "
                             "Please ensure the value of the parameter is "
                             "set before trying to access it.")
        return self.data.get(instance, self.default)

    def __set__(self, instance, value):
        self.validate(instance, value)
        self.data[instance] = value

    def __repr__(self):
        return "%s(default=%s, optional=%s, readonly=%s)" % (
            self.__class__.__name__,
            self.default,
            self.optional,
            self.readonly)

    @property
    def configurable(self):
        return self.default is not Unconfigurable

    def get_default(self, obj):
        return self._defaults.get(obj, self.default)

    def set_default(self, obj, value):
        if not self.configurable:
            raise ValueError("Parameter '%s' is not configurable" % self)
        self.validate(obj, value)
        self._defaults[obj] = value

    def del_default(self, obj):
        del self._defaults[obj]

    def validate(self, instance, value):
        if isinstance(value, DefaultType):
            raise ValueError("Default is not a valid value. To reset a "
                             "parameter, use `del`.")
        if self.readonly and instance in self.data:
            raise ValueError("Parameter is read-only; cannot be changed.")
        if not self.optional and value is None:
            raise ValueError("Parameter is not optional; cannot set to None")

    def equal(self, instance_a, instance_b):
        a = self.__get__(instance_a, None)
        b = self.__get__(instance_b, None)
        if self.equatable:
            # always use array_equal, in case one argument is an array
            return np.array_equal(a, b)
        else:
            return a is b

    def hashvalue(self, instance):
        """Returns a hashable value (`hash` can be called on the output)."""
        value = self.__get__(instance, None)
        if self.equatable:
            return value
        else:
            return id(value)


class ObsoleteParam(Parameter):
    """A parameter that is no longer supported."""

    def __init__(self, short_msg, url=None):
        self.short_msg = short_msg
        self.url = url
        super(ObsoleteParam, self).__init__(optional=True)

    def __get__(self, instance, type_):
        if instance is None:
            # Return self so default can be inspected
            return self
        self.raise_error()

    def validate(self, instance, value):
        if value is not Unconfigurable:
            # don't allow setting to anything other than unconfigurable default
            self.raise_error()

    def raise_error(self):
        raise ValueError("This parameter is no longer supported. %s%s" % (
            self.short_msg,
            "\nFor more information, please visit %s" % self.url
            if self.url is not None else ""))


class BoolParam(Parameter):
    equatable = True

    def validate(self, instance, boolean):
        if boolean is not None and not isinstance(boolean, bool):
            raise ValueError("Must be a boolean; got '%s'" % boolean)
        super(BoolParam, self).validate(instance, boolean)


class NumberParam(Parameter):
    equatable = True

    def __init__(self, default=Unconfigurable,
                 low=None, high=None, low_open=False, high_open=False,
                 optional=False, readonly=None):
        self.low = low
        self.high = high
        self.low_open = low_open
        self.high_open = high_open
        super(NumberParam, self).__init__(default, optional, readonly)

    def __set__(self, instance, value):
        if is_array(value) and value.shape == ():
            value = value.item()  # convert scalar array to Python object
        super(NumberParam, self).__set__(instance, value)

    def validate(self, instance, num):
        if num is not None:
            if not is_number(num):
                raise ValueError("Must be a number; got '%s'" % num)
            low_comp = 0 if self.low_open else -1
            if self.low is not None and compare(num, self.low) <= low_comp:
                raise ValueError("Value must be greater than %s%s (got %s)" % (
                    "" if self.low_open else "or equal to ", self.low, num))
            high_comp = 0 if self.high_open else 1
            if self.high is not None and compare(num, self.high) >= high_comp:
                raise ValueError("Value must be less than %s%s (got %s)" % (
                    "" if self.high_open else "or equal to ", self.high, num))
        super(NumberParam, self).validate(instance, num)


class IntParam(NumberParam):
    def validate(self, instance, num):
        if num is not None and not is_integer(num):
            raise ValueError("Must be an integer; got '%s'" % num)
        super(IntParam, self).validate(instance, num)


class StringParam(Parameter):
    equatable = True

    def validate(self, instance, string):
        if string is not None and not is_string(string):
            raise ValueError("Must be a string; got '%s'" % string)
        super(StringParam, self).validate(instance, string)


class EnumParam(StringParam):
    def __init__(self, default=Unconfigurable, values=(), lower=True,
                 optional=False, readonly=None):
        assert all(is_string(s) for s in values)
        if lower:
            values = tuple(s.lower() for s in values)
        value_set = set(values)
        assert len(values) == len(value_set)
        self.values = values
        self.value_set = value_set
        self.lower = lower
        super(EnumParam, self).__init__(default, optional, readonly)

    def __set__(self, instance, value):
        self.validate(instance, value)
        self.data[instance] = value.lower() if self.lower else value

    def validate(self, instance, string):
        super(EnumParam, self).validate(instance, string)
        string = string.lower() if self.lower else string
        if string not in self.value_set:
            raise ValueError("String %r must be one of %s"
                             % (string, list(self.values)))


class TupleParam(Parameter):
    def __init__(self, default=Unconfigurable, length=None,
                 optional=False, readonly=None):
        self.length = length
        super(TupleParam, self).__init__(default, optional, readonly)

    def __set__(self, instance, value):
        try:
            value = tuple(value)
        except TypeError:
            raise ValueError("Value must be castable to a tuple")
        super(TupleParam, self).__set__(instance, value)

    def validate(self, instance, value):
        if value is not None:
            if self.length is not None and len(value) != self.length:
                raise ValueError("Must be %d items (got %d)"
                                 % (self.length, len(value)))
        super(TupleParam, self).validate(instance, value)


class DictParam(Parameter):
    def validate(self, instance, dct):
        if dct is not None and not isinstance(dct, dict):
            raise ValueError("Must be a dictionary; got '%s'" % str(dct))
        super(DictParam, self).validate(instance, dct)


class NdarrayParam(Parameter):
    """Can be a NumPy ndarray, or something that can be coerced into one."""
    equatable = True

    def __init__(self, default=Unconfigurable, shape=None,
                 optional=False, readonly=None):
        assert shape is not None
        assert shape.count('...') <= 1, "Cannot have more than one ellipsis"
        self.shape = shape
        super(NdarrayParam, self).__init__(default, optional, readonly)

    def __set__(self, instance, ndarray):
        super(NdarrayParam, self).validate(instance, ndarray)
        if ndarray is not None:
            ndarray = self.validate(instance, ndarray)
        self.data[instance] = ndarray

    def validate(self, instance, ndarray):  # noqa: C901
        if isinstance(ndarray, np.ndarray):
            ndarray = ndarray.view()
        else:
            try:
                ndarray = np.array(ndarray, dtype=np.float64)
            except TypeError:
                raise ValueError("Must be a float NumPy array (got type '%s')"
                                 % ndarray.__class__.__name__)
        if self.readonly:
            ndarray.setflags(write=False)

        if '...' in self.shape:
            nfixed = len(self.shape) - 1
            n = ndarray.ndim - nfixed
            if n < 0:
                raise ValueError("ndarray must be at least %dD (got %dD)"
                                 % (nfixed, ndarray.ndim))

            i = self.shape.index('...')
            shape = list(self.shape[:i]) + (['*'] * n)
            if i < len(self.shape) - 1:
                shape.extend(self.shape[i+1:])
        else:
            shape = self.shape

        if ndarray.ndim != len(shape):
            raise ValueError("ndarray must be %dD (got %dD)"
                             % (len(shape), ndarray.ndim))

        for i, attr in enumerate(shape):
            assert is_integer(attr) or is_string(attr), (
                "shape can only be an int or str representing an attribute")
            if attr == '*':
                continue

            desired = attr if is_integer(attr) else getattr(instance, attr)

            if not is_integer(desired):
                raise ValueError("%s not yet initialized; cannot determine "
                                 "if shape is correct. Consider using a "
                                 "distribution instead." % attr)

            if ndarray.shape[i] != desired:
                raise ValueError("shape[%d] should be %d (got %d)"
                                 % (i, desired, ndarray.shape[i]))
        return ndarray

    def hashvalue(self, instance):
        return array_hash(self.__get__(instance, None))


FunctionInfo = collections.namedtuple('FunctionInfo', ['function', 'size'])


class FunctionParam(Parameter):
    def __set__(self, instance, function):
        size = (self.determine_size(instance, function)
                if callable(function) else None)
        function_info = FunctionInfo(function=function, size=size)
        super(FunctionParam, self).__set__(instance, function_info)

    def function_args(self, instance, function):
        return (np.zeros(1),)

    def determine_size(self, instance, function):
        args = self.function_args(instance, function)
        value, invoked = checked_call(function, *args)
        if not invoked:
            raise TypeError("function '%s' must accept a single "
                            "np.array argument" % function)
        return np.asarray(value).size

    def validate(self, instance, function_info):
        function = function_info.function
        if function is not None and not callable(function):
            raise ValueError("function '%s' must be callable" % function)
        super(FunctionParam, self).validate(instance, function)


class FrozenObject(object):
    def __init__(self):
        self._paramdict = dict(
            (k, v) for k, v in inspect.getmembers(self.__class__)
            if isinstance(v, Parameter))
        if not all(p.readonly for p in self._params):
            raise ValueError(
                "All parameters of a FrozenObject must be readonly")

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, ', '.join(
            "%s=%r" % (k, getattr(self, k)) for k in sorted(self._paramdict)))

    @property
    def _params(self):
        return itervalues(self._paramdict)

    def __eq__(self, other):
        if self is other:  # quick check for speed
            return True
        return self.__class__ == other.__class__ and all(
            p.equal(self, other) for p in self._params)

    def __hash__(self):
        return hash((self.__class__, tuple(
            p.hashvalue(self) for p in self._params)))

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop('_paramdict')  # do not pickle the param dict itself
        for k in self._paramdict:
            d[k] = getattr(self, k)

        return d

    def __setstate__(self, state):
        FrozenObject.__init__(self)  # set up the param dict
        for k in self._paramdict:
            setattr(self, k, state.pop(k))
        self.__dict__.update(state)
