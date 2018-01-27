import argparse
import getpass
import logging
import sys

from base import build_database, build_experiment
from space import Space

from sacred_commandline_options import (
    SelectOption, EnforceNewOption)


# To shut up pep8. We know they aren't used but we need to import them so that
# sacred registers them.
str(SelectOption)
str(EnforceNewOption)

DEBUG = "--debug" in sys.argv

logger = logging.getLogger()


walltime_limits = [10 * 60, 30 * 60, 60 * 60, 2 * 60 * 60, 12 * 60 * 60]

epochs = [10, 50, 200, 500]


def build_parser():

    parser = argparse.ArgumentParser(
        description="Exploration for paper \"Implicitly Natural\"")

    parser.add_argument("experiment_name", metavar="experiment-name",
                        help="Experiment name.")

    parser.add_argument("id", type=int,
                        help="ID of the experiment")

    parser.add_argument("--database-name", metavar="db-name",
                        help="Database name.")

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


def parse_args(argv):
    opt = build_parser().parse_args(argv)

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

    if len(opt.ports) == 1:
        opt.ports = opt.ports * len(opt.host_names)

    return opt


def main(argv=None):
    opt = parse_args(argv)
    database = build_database(opt)
    opt.profiles = []
    space = Space({}, opt)
    experiment = build_experiment(
        opt.experiment_name, "validation_accuracy", space, database,
        pool_size=1)

    trials = list(experiment.get_trials({"_id": {"$eq": opt.id}},
                                        evaluations=True))
    assert len(trials) == 1

    evaluation_trial = trials[0].get_evaluation()
    evaluation_trial.run()


if __name__ == "__main__":
    main()
