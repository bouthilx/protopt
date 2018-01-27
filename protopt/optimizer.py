import logging

import numpy

from skopt import Optimizer as SkoptOptimizer
from skopt import Space as SkoptSpace
from skopt.learning import GaussianProcessRegressor


logger = logging.getLogger()


class ValidatedSkoptOptimizer(SkoptOptimizer):
    def __init__(self, dimensions, validate_sample, **kwargs):
        super(ValidatedSkoptOptimizer, self).__init__(dimensions, **kwargs)

        self.validate_sample = validate_sample
        self.space = ValidatedSkoptSpace(self.space.dimensions,
                                         validate_sample)

    def ask(self, n_points=None, strategy="cl_min"):
        """Query point or multiple points at which objective should be evaluated.
        * `n_points` [int or None, default=None]:
            Number of points returned by the ask method.
            If the value is None, a single point to evaluate is returned.
            Otherwise a list of points to evaluate is returned of size
            n_points. This is useful if you can evaluate your objective in
            parallel, and thus obtain more objective function evaluations per
            unit of time.
        * `strategy` [string, default=`"cl_min"`]:
            Method to use to sample multiple points (see also `n_points`
            description). This parameter is ignored if n_points = None.
            Supported options are `"cl_min"`, `"cl_mean"` or `"cl_max"`.
            - If set to `"cl_min"`, then constant liar strtategy is used
               with lie objective value being minimum of observed objective
               values. `"cl_mean"` and `"cl_max"` means mean and max of values
               respectively. For details on this strategy see:
               https://hal.archives-ouvertes.fr/hal-00732512/document
               With this strategy a copy of optimizer is created, which is
               then asked for a point, and the point is told to the copy of
               optimizer with some fake objective (lie), the next point is
               asked from copy, it is also told to the copy with fake
               objective and so on. The type of lie defines different
               flavours of `cl_x` strategies.
        """
        if n_points is None:
            return self._ask()
        supported_strategies = ["cl_min", "cl_mean", "cl_max"]
        if not (isinstance(n_points, int) and n_points > 0):
            raise ValueError(
                "n_points should be int > 0, got " + str(n_points)
            )
        if strategy not in supported_strategies:
            raise ValueError(
                "Expected parallel_strategy to be one of " +
                str(supported_strategies) + ", " + "got %s" % strategy
            )
        # Caching the result with n_points not None. If some new parameters
        # are provided to the ask, the cache_ is not used.
        if (n_points, strategy) in self.cache_:
            return self.cache_[(n_points, strategy)]

        # Copy of the optimizer is made in order to manage the
        # deletion of points with "lie" objective (the copy of
        # optimizer is simply discarded)
        opt = self.copy(
            random_state=self.rng.randint(0, numpy.iinfo(numpy.int32).max))

        trashed = 0
        X = []
        while len(X) < n_points:
            x = opt.ask()
            X.append(x)

            if not self.validate_sample(x):
                y_lie = 0.0  # Invalid sample
                # Tell the main optimizer too
                # Nah, we retrain from scratch each time anyway
                # self.tell(x, y_lie)
                X.pop(-1)
                trashed += 1
            elif strategy == "cl_min":
                y_lie = numpy.min(opt.yi) if opt.yi else 0.0  # CL-min lie
            elif strategy == "cl_mean":
                y_lie = numpy.mean(opt.yi) if opt.yi else 0.0  # CL-mean lie
            else:
                y_lie = numpy.max(opt.yi) if opt.yi else 0.0  # CL-max lie

            opt.tell(x, y_lie)  # lie to the optimizer

        logger.info("Optimizer.ask() trashed %d invalid samples" % trashed)

        self.cache_ = {(n_points, strategy): X}  # cache_ the result
        return X

    def copy(self, random_state=None):
        """Create a shallow copy of an instance of the optimizer.

        Parameters
        ----------
        * `random_state` [int, RandomState instance, or None (default)]:
            Set the random state of the copy.
        """

        optimizer = ValidatedSkoptOptimizer(
            dimensions=self.space.dimensions,
            validate_sample=self.validate_sample,
            base_estimator=self.base_estimator_,
            n_initial_points=self.n_initial_points_,
            acq_func=self.acq_func,
            acq_optimizer=self.acq_optimizer,
            acq_func_kwargs=self.acq_func_kwargs,
            acq_optimizer_kwargs=self.acq_optimizer_kwargs,
            random_state=random_state,
        )

        if hasattr(self, "gains_"):
            optimizer.gains_ = numpy.copy(self.gains_)

        if self.Xi:
            optimizer.tell(self.Xi, self.yi)

        return optimizer


class ValidatedSkoptSpace(SkoptSpace):
    def __init__(self, dimensions, validate_sample=None):
        super(ValidatedSkoptSpace, self).__init__(dimensions)

        self.validate_sample = validate_sample

    def rvs(self, n_samples=1, random_state=None):
        rows, trashed = self._rvs(n_samples=1, random_state=None)

        logger.info("Space.rvs() trashed %d invalid samples" % trashed)

        return rows

    def _rvs(self, n_samples=1, random_state=None, trashed=0):
        # Balance between filtering a sampling and adding knowledge in
        # Optimizer about invalid points
        # (always adding knowledge is expensive)
        if trashed > n_samples:
            rows = super(ValidatedSkoptSpace, self).rvs(
                n_samples, random_state)
            return rows, trashed

        rows = super(ValidatedSkoptSpace, self).rvs(n_samples * 10,
                                                    random_state)

        rows = filter(self.validate_sample, rows)
        trashed += n_samples * 10 - len(rows)

        if len(rows) < n_samples:
            try:
                new_rows, trashed = self._rvs(n_samples, random_state, trashed)
                rows = rows + new_rows
            except Exception:  # RecursionError:
                raise ValueError(
                    'Dimensions and validate_sample are incompatible.')

        return rows[:n_samples], trashed


class Optimizer(object):
    def __init__(self, pool_size, space):
        self.pool_size = pool_size
        self.space = space

    def _build_optimizer(self, **kwargs):
        print "Building optimizer"
        optimizer = ValidatedSkoptOptimizer(
            base_estimator=GaussianProcessRegressor(**kwargs),
            dimensions=self.space.get_spaces().values(),
            validate_sample=self.space.get_validate_sample_fct()
        )

        return optimizer

    def _get_trained_optimizer(self, x, y, tries=10, **kwargs):
        optimizer = self._build_optimizer(**kwargs)

        if x:
            optimizer.tell(x, y)

        return optimizer

    def _get_random_candidate(self):
        optimizer = self._build_optimizer()
        return optimizer.space.rvs(n_samples=self.pool_size)

    def _get_bayesian_opt_candidate(self, x, y, maximum_tries):
        alpha = 0.
        number_of_tries = 0
        while True:
            try:
                logger.info("Training optimizer on %d points" % len(x))
                optimizer = self._get_trained_optimizer(x, y, alpha=alpha)
                logging.info("Sampling %d new points" % self.pool_size)
                new_candidates = optimizer.ask(n_points=self.pool_size)
                break
            except numpy.linalg.linalg.LinAlgError as e:
                logging.warning("Optimizer failed on try %d: %s" %
                                (number_of_tries, str(e)))
                number_of_tries += 1
                alpha = (alpha + 0.1) * 1.1
                logging.info("Setting alpha to %f" % alpha)
                if number_of_tries >= maximum_tries:
                    raise

        return new_candidates

    def get_new_candidates(self, x, y, maximum_tries=10):
        if numpy.random.uniform() < 0.25:
            new_candidates = self._get_random_candidate()
        else:
            new_candidates = self._get_bayesian_opt_candidate(
                x, y, maximum_tries)

        # Remove duplicates
        past_candidates = set(tuple(c) for c in x)
        unique_candidates = list(set(tuple(n_c) for n_c in new_candidates
                                     if tuple(n_c) not in past_candidates))
        logger.info("Optimizer sampled %d unique candidates. "
                    "(There was %d duplicates)" %
                    (len(unique_candidates),
                     len(new_candidates) - len(unique_candidates)))
        return unique_candidates
