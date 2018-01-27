import six
import sys


try:
    from sacred.utils import SacredInterrupt
except ImportError:
    SacredInterrupt = RuntimeError


def options_to_dict(options):
    return options.__dict__


def dict_to_options(setting, options):

    for key, value in setting.iteritems():
        assert hasattr(options, key), (
            "Setting has a key which is not part of the "
            "options: %s" % key)
        setattr(options, key, value)

    return options


class Interrupt(SacredInterrupt):
    STATUS = "INTERRUPTED"


class TimeoutInterrupt(Interrupt):
    pass


class SacredSelectionError(RuntimeError):
    # We do not set a status because there is nothing to save in the run, it
    # just didn't start.
    pass


class ClusterProblem(Interrupt):
    STATUS = "CLUSTER_PROBLEM"


class InvalidConfiguration(Interrupt):
    STATUS = "INVALID"


class NumericalStabilityError(RuntimeError):
    # We do not set a status because it is intended to be handled by the user
    # script.
    pass


class VerboseNumericalStabilityError(object):
    def __init__(self, tensor):
        self.tensor = tensor

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if (exc_type is RuntimeError and
                "value cannot be converted" in str(exc_value)):
            message = str(exc_value) + ":"
            for power in [1, 2]:
                for fct in ["min", "mean", "max"]:
                    try:
                        value = "%f" % getattr(self.tensor ** power, fct)()
                    except RuntimeError:
                        value = "Error (NaN)"
                    message += "\n%s(%d)=%s" % (fct, power, value)

            six.reraise(NumericalStabilityError,
                        NumericalStabilityError(message),
                        sys.exc_info()[2])


def _add_attribute(value, default, separator=""):
    if value or default:
        value = value if value else default
        return separator + str(value)
    else:
        return ""


def get_mongodb_url(hosts, user=None, password=None, db=None,
                    **kwargs):

    mongodb_url = "mongodb://"
    mongodb_url += _add_attribute(user, None)
    mongodb_url += _add_attribute(password, None, ":")

    hosts = ",".join(["%s:%d" % (host, port) for (host, port) in hosts])

    mongodb_url += _add_attribute(
        hosts, None, "@" if mongodb_url[-1] != "/" else "")

    mongodb_url += _add_attribute(db, None, "/")

    options = "&".join("%s=%s" % (k, v) for (k, v) in kwargs.iteritems())

    mongodb_url += _add_attribute(
        options, None, "/?" if not (db or None) else "?")

    return mongodb_url
