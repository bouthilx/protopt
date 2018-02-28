import copy
import logging
import os
import sys

import status

from sacred import Experiment


DEBUG = "--debug" in sys.argv

logger = logging.getLogger()


class Trial(object):
    def __init__(self, setting, experiment, row=None):
        self.setting = setting
        self.experiment = experiment
        self.row = row

    @property
    def id(self):
        if self.row is None:
            return None

        return self.row["_id"]

    @property
    def result(self):
        if self.row is None:
            return None

        return self.experiment.get_result(self)

    @property
    def status(self):
        if self.row is None:
            return None

        return self.row["status"]

    @property
    def metrics(self):
        if self.row is None:
            return {}

        if "metrics" not in self.row:
            return {}
            return self._metrics_backward_compatibility()

        metrics = {}
        for metric_name, scalar_metrics in self.row["metrics"].iteritems():
            # steps = metrics_row["steps"]
            # key_template = "%s.%s"

            # Metrics have
            # _id
            # name
            # steps
            # values
            # timestamps

            steps = scalar_metrics["steps"]
            values = scalar_metrics["values"]
            timestamps = scalar_metrics["timestamps"]

            metrics[metric_name] = dict(
                epoch=dict(zip(steps, values)),
                timestamp=dict(zip(timestamps, values)))

        # units = self.row["metrics"]["units"]
        # scalars = self.row["metrics"]["scalars"]
        # metrics = {}

        # for metric_name, metric in scalars.iteritems():
        #     for unit_name in metric["units"]:
        #         metric_unit_name = "%s.%s" % (metric_name, unit_name)
        #         steps = units[unit_name]
        #         values = metric["values"]
        #         metrics[metric_unit_name] = dict(
        #             zip((str(s) for s in steps), values))

        return metrics

    def _metrics_backward_compatibility(self):

        metrics = {}
        metrics_rows = self.experiment.database.mongo_observer.metrics.find(
            {"run_id": self.id})

        logger.debug("Found %d metrics for trial %d" %
                     (metrics_rows.count(), self.id))

        for i, metrics_row in enumerate(list(metrics_rows)):
            logger.debug("Parsing %d-th metric" % i)
            # steps = metrics_row["steps"]
            # key_template = "%s.%s"

            # Metrics have
            # _id
            # name
            # steps
            # values
            # timestamps

            name = metrics_row["name"]
            steps = metrics_row["steps"]
            values = metrics_row["values"]
            metrics[name] = dict(zip((str(s) for s in steps), values))

        return metrics

    def is_completed(self):
        return self.status in status.COMPLETED

    def is_runnable(self):
        return self.status in status.RUNNABLE

    def is_queued(self):
        return self.status in status.QUEUED

    def run(self):
        if not self.is_runnable():
            raise RuntimeError("Trial is not runnable")

        self._run({"--select": self.id})

    # def get_evaluation(self):

    #     evaluation_trial = Trial(copy.deepcopy(self.setting),
    #                              self.experiment)
    #     evaluation_trial.setting["validate"] = False
    #     max_epochs_reached = self.metrics["validation_accuracy"]["steps"][-1]
    #     evaluation_trial.setting["epochs"] = max_epochs_reached
    #     evaluation_trial.queue()

    #     return evaluation_trial

    def _run(self, run_options):
        # Make sure there is no overwrite left in the observer
        self.experiment.database.mongo_observer.overwrite = None

        ex = Experiment(self.experiment.name)
        ex.main(self.experiment.fct)
        ex.add_config(self.experiment.default_setting)
        config_updates = copy.copy(self.setting)

        options = {
            "--queue": False,
            "--unobserved": False,
            "--enforce_clean": False}  # not DEBUG}

        options.update(run_options)

        # Means we are going to run a job from scratch (not resuming)
        # We set dataroot and save specific to current cluster
        is_a_queued_trial = self.row is not None and "host" not in self.row

        if not options["--queue"] and is_a_queued_trial:
            config_updates["data_path"] = os.environ.get("DATA_PATH", ".")

            sorted_profiles = [name for name, _
                               in self.experiment.space.iter_profiles()
                               if name is not None]

            config_updates["save_path"] = os.path.join(*(
                [self.experiment.dir_path] +
                sorted_profiles))

            if "tensorboard" in self.experiment.default_setting:
                config_updates["tensorboard"] = os.path.join(*(
                    [self.experiment.dir_path, "logs"] +
                    sorted_profiles))

        ex.observers.append(self.experiment.database.mongo_observer)

        logger.debug("Executing experiment.run")
        ex.run(config_updates=config_updates, named_configs=(),
               meta_info=None, options=options)

        return ex

    def queue(self):
        # Clean
        self.setting["data_path"] = (
            self.experiment.default_setting["data_path"])
        self.setting["save_path"] = (
            self.experiment.default_setting["save_path"])

        self._run({"--queue": True})
        self.row = self.experiment.database.mongo_observer.run_entry
        self.setting = self.row["config"]
