import argparse
import getpass
import logging
import os
import random
import sys

from database import Database
from experiment import Experiment
from optimizer import Optimizer
from utils import SacredSelectionError, Interrupt, ClusterProblem
from sacred_commandline_options import SelectOption, EnforceNewOption
from wrapper import fake_fct_signature


# To shut up pep8. We know they aren't used but we need to import them so that
# sacred registers them.
str(SelectOption)
str(EnforceNewOption)

DEBUG = "--debug" in sys.argv

logger = logging.getLogger()


def build_parser(models):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", choices=models.keys(),
        help=("Which kind of model to meta-optimize. Choices are %s" %
              str(models.keys())))

    parser.add_argument("--experiment-name", metavar="experiment-name",
                        help="Experiment name. Default is cesar_{model}")

    parser.add_argument(
        "--profiles", nargs="*", default=[],
        help=("Select profiles to limit possible hyper-parameters."))

    parser.add_argument(
        "--validate-on",
        default="validation_accuracy",
        help=("Which metric to use to evaluate trials. Default is "
              "'validation_accuracy.epoch' which means validation accuracy"
              "of the last epoch."))

    parser.add_argument("--database-name", metavar="db-name",
                        help="Database name.")

    parser.add_argument(
        "--patience", type=int, default=5,
        help=("How many tries to run experiments before sampling "
              "new candidates"))

    parser.add_argument(
        "--pool-size", type=int, default=10,
        help=("How many candidates to sample at the same time. "
              "This will not strictly limit the number of workers "
              "to --pool-size but should keep their number around this "
              "approximately."))

    parser.add_argument(
        "--host-names", default=["localhost"], nargs="*",
        help="Host where the mongoDB database is to store configurations and "
             "results")

    parser.add_argument(
        "--ports", default=[27017], nargs="*", type=int,
        help="Host for the mongodb database")

    parser.add_argument(
        "--ssl", action="store_true",
        help="")

    parser.add_argument(
        "--ssl-ca-file",
        help="")

    parser.add_argument(
        "--replica-set",
        help="")

    parser.add_argument(
        "--auth-source",
        help="")

    parser.add_argument(
        "--user-name", default=getpass.getuser(),
        help="User name for the mongoDB database.")

    parser.add_argument(
        "--password", default="",
        help="Password for the mongoDB database.")

    # Debugging options
    parser.add_argument('--debug', action="store_true",
                        help='Use a small subset of settings to speed up '
                             'exploration during debugging')

    parser.add_argument('--gpu-id', type=int, default=0)

    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help="Print informations about the process.\n"
             "     -v: INFO\n"
             "     -vv: DEBUG")

    parser.add_argument(
        '-p', '--verbose-process', action='count', default=0,
        help="Print informations about the experiment.\n"
             "     -p: INFO\n"
             "     -pp: DEBUG")

    return parser


def parse_args(models, argv):
    opt = build_parser(models).parse_args(argv)

    if opt.debug:
        opt.verbose = 2

    if opt.verbose == 0:
        logging.basicConfig(level=logging.WARNING)
        logger.setLevel(level=logging.WARNING)
    elif opt.verbose == 1:
        logging.basicConfig(level=logging.INFO)
        logger.setLevel(level=logging.INFO)
    elif opt.verbose == 2:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(level=logging.DEBUG)

    if opt.experiment_name is None:
        opt.experiment_name = "cesar_%s" % opt.model.replace("-", "_")

    if len(opt.ports) == 1:
        opt.ports = opt.ports * len(opt.host_names)

    return opt


def build_database(opt):
    return Database(opt.database_name, opt.host_names, opt.ports,
                    opt.user_name, opt.password, opt.ssl, opt.ssl_ca_file,
                    opt.replica_set, opt.auth_source)


def build_experiment(name, validate_on, space, database, pool_size):
    optimizer = Optimizer(pool_size, space)

    dir_path = os.path.join(os.getcwd(), name)

    experiment = Experiment(
        name=name, dir_path=dir_path, fct=fake_fct_signature,
        validate_on=validate_on, space=space,
        optimizer=optimizer, database=database)

    return experiment


def main_loop(name, validate_on, space, database, pool_size):

    experiment = build_experiment(name, validate_on, space, database,
                                  pool_size)

    resilience = 10

    while resilience > 0:

        trials = experiment.get_runnable_trials()
        skip = random.randint(1, 5)
        iter_trials = iter(trials)
        logger.debug("Skipping %d trials" % skip)
        for i in xrange(skip):
            try:
                trial = next(iter_trials)
                logger.debug("Skipping %d-th with id %d" % (i, trial.id))
            except StopIteration:
                # TODO FINISH
                if i == 0:
                    raise RuntimeError("Experiment could not return any "
                                       "runnable trials")

        logger.debug("Selected %d-th trial with id %d" % (i, trial.id))
        # Try to launch it. If another process select it between
        # select_random_config() and run(), run() will raise a ValueError
        try:
            trial.run()
        except SacredSelectionError as e:
            logger.info("Failed to launch %d. It could be because of a race "
                        "condition with another worker.\n %s" %
                        (trial.id, str(e)))
            experiment.exclude(trial)
        except ClusterProblem as e:
            logger.info("Failed to launch %d because of a problem "
                        "on the cluster:\n %s" % (trial.id, str(e)))
            experiment.exclude(trial)
        except Interrupt as e:
            if "File not available" in str(e):
                logger.info("Wrong cluster, skipped job %d" % trial.id)
                experiment.exclude(trial)
            else:
                raise e
        except BaseException as e:
            logger.error(str(e))
            resilience -= 1
            logger.info("Resilience now at %d" % resilience)

        # If killed by timeout, the worker will be rescheduler by
        # SmartDispatch and it will look again for a job in the db, which
        # could be the one paused that resume
