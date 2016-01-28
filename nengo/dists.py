from __future__ import absolute_import
import numpy as np

from nengo.params import (BoolParam, IntParam, NdarrayParam, NumberParam,
                          Parameter, Unconfigurable, FrozenObject)
import nengo.utils.numpy as npext


class Distribution(FrozenObject):
    """A base class for probability distributions.

    The only thing that a probabilities distribution need to define is a
    ``sample`` function. This base class ensures that all distributions
    accept the same arguments for the sample function.

    """
    def sample(self, n, d=None, rng=np.random):
        """Samples the distribution.

        Parameters
        ----------
        n : int
            Number samples to take.
        d : int or None, optional
            The number of dimensions to return. If this is an int, the return
            value will be of shape ``(n, d)``. If None (default), the return
            value will be of shape ``(n,)``.
        rng : RandomState, optional
            Random number generator state.

        Returns
        -------
        ndarray
            Samples as a 1d or 2d array depending on ``d``. The second
            dimension enumerates the dimensions of the process.
        """
        raise NotImplementedError("Distributions should implement sample.")


class PDF(Distribution):
    """An arbitrary distribution from a PDF.

    Parameters
    ----------
    x : vector_like (n,)
        Values of the points to sample from (interpolated).
    y : vector_like (n,)
        Probabilities of the `x` points.
    """
    x = NdarrayParam(shape='*')
    p = NdarrayParam(shape='*')

    def __init__(self, x, p):
        super(PDF, self).__init__()

        psum = np.sum(p)
        if np.abs(psum - 1) > 1e-8:
            raise ValueError("PDF must sum to one (sums to %f)" % psum)

        self.x = x
        self.p = p
        if len(self.x) != len(self.p):
            raise ValueError("`x` and `p` must be the same length")

        # make cumsum = [0] + cumsum, cdf = 0.5 * (cumsum[:-1] + cumsum[1:])
        cumsum = np.cumsum(p)
        cumsum *= 0.5
        cumsum[1:] = cumsum[:-1] + cumsum[1:]
        self.cdf = cumsum

    def __repr__(self):
        return "PDF(x=%r, p=%r)" % (self.x, self.p)

    def sample(self, n, d=None, rng=np.random):
        shape = (n,) if d is None else (n, d)
        return np.interp(rng.uniform(size=shape), self.cdf, self.x)


class Uniform(Distribution):
    """A uniform distribution.

    It's equally likely to get any scalar between ``low`` and ``high``.

    Note that the order of ``low`` and ``high`` doesn't matter;
    if ``low < high`` this will still work, and ``low`` will still
    be a closed interval while ``high`` is open.

    Parameters
    ----------
    low : Number
        The closed lower bound of the uniform distribution; samples >= low
    high : Number
        The open upper bound of the uniform distribution; samples < high
    integer : boolean, optional
        If true, sample from a uniform distribution of integers. In this case,
        low and high should be integers.
    """
    low = NumberParam()
    high = NumberParam()
    integer = BoolParam()

    def __init__(self, low, high, integer=False):
        super(Uniform, self).__init__()
        self.low = low
        self.high = high
        self.integer = integer

    def __repr__(self):
        return "Uniform(low=%r, high=%r%s)" % (
            self.low, self.high, ", integer=True" if self.integer else "")

    def sample(self, n, d=None, rng=np.random):
        shape = (n,) if d is None else (n, d)
        if self.integer:
            return rng.randint(low=self.low, high=self.high, size=shape)
        else:
            return rng.uniform(low=self.low, high=self.high, size=shape)


class Gaussian(Distribution):
    """A Gaussian distribution.

    This represents a bell-curve centred at ``mean`` and with
    spread represented by the standard deviation, ``std``.

    Parameters
    ----------
    mean : Number
        The mean of the Gaussian.
    std : Number
        The standard deviation of the Gaussian.

    Raises
    ------
    ValueError if ``std <= 0``

    """
    mean = NumberParam()
    std = NumberParam(low=0, low_open=True)

    def __init__(self, mean, std):
        super(Gaussian, self).__init__()
        self.mean = mean
        self.std = std

    def __repr__(self):
        return "Gaussian(mean=%r, std=%r)" % (self.mean, self.std)

    def sample(self, n, d=None, rng=np.random):
        shape = (n,) if d is None else (n, d)
        return rng.normal(loc=self.mean, scale=self.std, size=shape)


class UniformHypersphere(Distribution):
    """Distributions over an n-dimensional unit hypersphere.

    Parameters
    ----------
    surface : bool
        Whether sample points should be distributed uniformly
        over the surface of the hyperphere (True),
        or within the hypersphere (False).
        Default: False

    """
    surface = BoolParam()

    def __init__(self, surface=False):
        super(UniformHypersphere, self).__init__()
        self.surface = surface

    def __repr__(self):
        return "UniformHypersphere(%s)" % (
            "surface=True" if self.surface else "")

    def sample(self, n, d, rng=np.random):
        if d is None or d < 1:  # check this, since other dists allow d = None
            raise ValueError("Dimensions must be a positive integer")

        samples = rng.randn(n, d)
        samples /= npext.norm(samples, axis=1, keepdims=True)

        if self.surface:
            return samples

        # Generate magnitudes for vectors from uniform distribution.
        # The (1 / d) exponent ensures that samples are uniformly distributed
        # in n-space and not all bunched up at the centre of the sphere.
        samples *= rng.rand(n, 1) ** (1.0 / d)

        return samples


class Choice(Distribution):
    """Discrete distribution across a set of possible values.

    The same as `numpy.random.choice`, except can take vector or matrix values
    for the choices.

    Parameters
    ----------
    options : array_like (N, ...)
        The options (choices) to choose between. The choice is always done
        along the first axis, so if `options` is a matrix, the options are
        the rows of that matrix.
    weights : array_like (N,) (optional)
        Weights controlling the probability of selecting each option. Will
        automatically be normalized. Defaults to a uniform distribution.
    """
    options = NdarrayParam(shape=('*', '...'))
    weights = NdarrayParam(shape=('*'), optional=True)

    def __init__(self, options, weights=None):
        super(Choice, self).__init__()
        self.options = options
        self.weights = weights

        weights = (np.ones(len(self.options)) if self.weights is None else
                   self.weights)
        if len(weights) != len(self.options):
            raise ValueError(
                "Number of weights (%d) must match number of options (%d)"
                % (len(weights), len(self.options)))
        if not all(weights >= 0):
            raise ValueError("All weights must be non-negative")
        total = float(weights.sum())
        if total <= 0:
            raise ValueError(
                "Sum of weights must be positive (got %f)" % total)
        self.p = weights / total

    def __repr__(self):
        return "Choice(options=%r%s)" % (
            self.options,
            "" if self.weights is None else ", weights=%r" % self.weights)

    def sample(self, n, d=None, rng=np.random):
        if d is not None and np.prod(self.options.shape[1:]) != d:
            raise ValueError("Options must be of dimensionality %d "
                             "(got %d)" % (d, np.prod(self.options.shape[1:])))

        i = np.searchsorted(np.cumsum(self.p), rng.rand(n))
        return self.options[i]


class SqrtBeta(Distribution):
    """Distribution of the square root of a Beta distributed random variable.

    Given `n + m` dimensional random unit vectors, the length of subvectors
    with `m` elements will be distributed according to this distribution.

    Parameters
    ----------
    n, m : Number
        Shape parameters of the distribution.

    See also
    --------
    SubvectorLength
    """
    n = IntParam(low=0)
    m = IntParam(low=0)

    def __init__(self, n, m=1):
        super(SqrtBeta, self).__init__()
        self.n = n
        self.m = m

    def __repr__(self):
        return "%s(n=%r, m=%r)" % (self.__class__.__name__, self.n, self.m)

    def sample(self, num, d=None, rng=np.random):
        shape = (num,) if d is None else (num, d)
        return np.sqrt(rng.beta(self.m / 2.0, self.n / 2.0, size=shape))

    def pdf(self, x):
        """Probability distribution function.

        Requires Scipy.

        Parameters
        ----------
        x : ndarray
            Evaluation points in [0, 1].

        Returns
        -------
        ndarray
            Probability density at `x`.
        """
        from scipy.special import beta
        return (2 / beta(self.m / 2.0, self.n / 2.0) * x ** (self.m - 1) *
                (1 - x * x) ** (self.n / 2.0 - 1))

    def cdf(self, x):
        """Cumulative distribution function.

        Requires Scipy.

        Parameters
        ----------
        x : ndarray
            Evaluation points in [0, 1].

        Returns
        -------
        ndarray
            Probability that `X <= x`.
        """
        from scipy.special import betainc
        sq_x = x * x
        return np.where(
            sq_x < 1., betainc(self.m / 2.0, self.n / 2.0, sq_x),
            np.ones_like(x))


class SubvectorLength(SqrtBeta):
    """Distribution of the length of a subvectors of a unit vector.

    Parameters
    ----------
    dimensions : int
        Dimensionality of the complete unit vector.
    subdimensions : int, optional
        Dimensionality of the subvector.

    See also
    --------
    SqrtBeta
    """
    def __init__(self, dimensions, subdimensions=1):
        super(SubvectorLength, self).__init__(
            dimensions - subdimensions, subdimensions)

    def __repr__(self):
        return "%s(%r, subdimensions=%r)" % (
            self.__class__.__name__, self.n + self.m, self.m)


class DistributionParam(Parameter):
    """A Distribution."""
    equatable = True

    def validate(self, instance, dist):
        if dist is not None and not isinstance(dist, Distribution):
            raise ValueError("'%s' is not a Distribution type" % dist)
        super(DistributionParam, self).validate(instance, dist)


class DistOrArrayParam(NdarrayParam):
    """Can be a Distribution or samples from a distribution."""

    def __init__(self, default=Unconfigurable, sample_shape=None,
                 optional=False, readonly=None):
        super(DistOrArrayParam, self).__init__(
            default, sample_shape, optional, readonly)

    def validate(self, instance, distorarray):
        if isinstance(distorarray, Distribution):
            Parameter.validate(self, instance, distorarray)
            return distorarray
        else:
            return NdarrayParam.validate(self, instance, distorarray)
