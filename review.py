import scipy
from scipy import stats
import numpy as np
from scipy import integrate

class ReviewModel():

  class Dirac_gen(stats.rv_continuous):
    """Dirac delta distribution, necessary as a limiting distribution."""
    def _rvs(self, *x, **y):
      return np.zeros(self._size);

    def _cdf(self, x):
      return 1.*(x >= 0);

    def _pdf(self, x):
      if (x == 0):
        return np.inf;
      else:
        return 0;

  Dirac = Dirac_gen(name='Dirac');

  def __init__(self, p_fourstar_threshold, mu_value, sigma2_err, sigma2_value=1):
    """ Initialize review model with indicate parameter values."""
    mu_err = -sigma2_err/2.0;
    self.p_fourstar_threshold = p_fourstar_threshold;
    self.mu_value = mu_value;
    self.sigma2_err = sigma2_err;
    self.sigma2_value = sigma2_value;
    if (sigma2_value < 0):
      raise ValueError('Cannot accept negative sigma2');

    if (sigma2_err == 0):
      self.err_dist = ReviewModel.Dirac(loc=1)
    elif (sigma2_err > 0):
      self.err_dist = stats.lognorm(scale=np.exp(mu_err), s=np.sqrt(sigma2_err));
    elif (sigma2_err < 0):
      raise ValueError('Cannot accept negative sigma2');

    self.value_dist = stats.lognorm(scale=np.exp(mu_value), s=np.sqrt(sigma2_value));
    self.perceived_dist = stats.lognorm(scale=np.exp(mu_value + mu_err), s=np.sqrt(sigma2_value + sigma2_err));
    self.prob_over_fourstar = 1 - self.perceived_dist.cdf(p_fourstar_threshold);
    self.prob_under_fourstar = self.perceived_dist.cdf(p_fourstar_threshold);

  @staticmethod
  def MuValueFromFourStarProportion(pp_fourstar, p_fourstar_threshold, sigma2_err, sigma2_value=1):
    """ Initialize review model where the mu_value is inferred from the pp_fourstar, given the
        `p_fourstar_threshold`, `sigma2_err` and `sigma2_value`. """
    mu_value = np.log(p_fourstar_threshold) \
               - np.sqrt(2.0*(sigma2_value + sigma2_err))*scipy.special.erfinv(1.0 - 2.0*pp_fourstar) \
               + sigma2_err/2.0;
    if np.isposinf(mu_value):
      mu_value = 10;
    elif np.isneginf(mu_value):
      mu_value = -10;
    return ReviewModel(p_fourstar_threshold, mu_value, sigma2_err, sigma2_value);

  def cond_pdf(self, x, fourstar=True):
    """ Conditional probability that a publication has a certain value, given it is awarded four stars or not, i.e. Pr(value | 4*). """
    if fourstar:
      return (1 - self.err_dist.cdf(self.p_fourstar_threshold/x))*self.value_dist.pdf(x)/self.prob_over_fourstar;
    else:
      return self.err_dist.cdf(self.p_fourstar_threshold/x)*self.value_dist.pdf(x)/self.prob_under_fourstar;

  def accuracy(self, fourstar=True):
    """ The accuracy of peer review, defined as the probability that a four star publication would again be awarded four stars. """
    def acc_f(x):
      p = self.cond_pdf(x, fourstar)
      if fourstar:
        p *= 1 - self.err_dist.cdf(self.p_fourstar_threshold/x);
      else:
        p *= self.err_dist.cdf(self.p_fourstar_threshold/x);
      return p;

    return integrate.quad(lambda x: acc_f(x), a=0, b=np.inf)[0];

  def cond_sample_value(self, sample_size, fourstar=True):
    """ Return a sample of values of size ``sample_size'' conditional on whether they are rated four star or not. """
    value_sample = [];
    while len(value_sample) < sample_size:
      size = np.ceil(sample_size - len(value_sample));
      if fourstar and self.prob_over_fourstar > 0:
          size /= self.prob_over_fourstar;
      elif self.prob_under_fourstar > 0:
          size /= self.prob_under_fourstar;

      v = self.value_dist.rvs(size=int(size));
      e = self.err_dist.rvs(size=int(size));
      p = v*e;

      if fourstar:
        value_sample.extend( v[p > self.p_fourstar_threshold][:sample_size - len(value_sample)] );
      else:
        value_sample.extend( v[p < self.p_fourstar_threshold][:sample_size - len(value_sample)] );

    return value_sample;

  def cond_sample_fourstar(self, sample_size, fourstar=True):
    """
    Return a fourstar sample of size ``sample_size'' conditional on whether they are rated fourstar.
    The return value is ``True'' whenever the sampled perceived value is larger than the threshold.
    """
    value_sample = self.cond_sample_value(sample_size, fourstar=fourstar);
    err = self.err_dist.rvs(size=sample_size);
    return value_sample*err > self.p_fourstar_threshold;
