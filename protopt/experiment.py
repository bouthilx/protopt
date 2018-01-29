import itertools
import logging

import numpy

import pymongo

import smartdispatch.utils

from sacred import host_info_getter

import status
from trials import Trial


logger = logging.getLogger()

CLUSTER_NAME = smartdispatch.utils.detect_cluster()


@host_info_getter
def cluster():
    return CLUSTER_NAME


class Experiment(object):

    def __init__(self, name, dir_path, fct, validate_on, space, optimizer,
                 database):
        self.name = name
        self.dir_path = dir_path
        self.fct = fct
        self.validate_on = validate_on
        self.space = space
        self.optimizer = optimizer
        self.database = database
        self.excluded_trials = set()

    def _build_trial(self, row):
        config = row["config"]

        # spaces *MUST* be an OrderedDict
        # hp_set = self.space.dict_to_set(config)

        self.space.force_options(config)
        trial = Trial(config, self, row=row if "_id" in row else None)

        return trial

    @property
    def default_setting(self):
        return self.space.get_default()

    def get_result(self, trial):
        # result after n seconds
        # trial.metrics["validation_accuracy.walltime"][3600]

        # result after n epochs
        # trial.metrics["validation_accuracy.epoch"][10]

        keys = self.validate_on.split(".")
        if len(keys) > 2:
            metric_name = ".".join(keys[:-2])
            unit_name = keys[-2]
            step = keys[-1]
        elif len(keys) == 2:
            metric_name = ".".join(keys[:-1])
            unit_name = keys[-1]
            step = None
        else:
            metric_name = self.validate_on
            unit_name = "epoch"
            step = None

        metrics = trial.metrics

        # InvalidConfiguration
        if len(metrics) == 0 and trial.is_completed():
            assert trial.row.get("result", 0.) == 0.
            result = 0.
        # Valid configuration
        else:
            try:
                unit_metrics = metrics[metric_name]
            except KeyError as e:
                raise type(e)(str(e) + (". Can be one of: %s" %
                                        str(sorted(metrics.keys()))))
            try:
                result = unit_metrics[unit_name]
            except KeyError as e:
                raise type(e)(str(e) + (". Can be one of: %s" %
                                        str(sorted(unit_metrics.keys()))))

        # WAIT Those are not list, they are dictionnaries!
        if isinstance(result, dict) and step is not None:
            # TODO
            # Get the item with key step or take the first smaller or equal
            # value in the sorted list.
            if step in result:
                step_value = step
                result = result[step]
            else:
                sorted_keys = sorted(result.keys())
                for step_value, next_step in zip(sorted_keys, sorted_keys[1:]):
                    if next_step > step:
                        break

                # If best step found
                if next_step > step:
                    result = result[step_value]
                # Otherwise use last step
                else:
                    result = result[next_step]
                    step_value = next_step
        elif isinstance(result, dict):
            last_step_key = sorted(result.keys())[-1]
            result = result[last_step_key]
            step_value = last_step_key

        if not isinstance(result, float):
            raise ValueError(
                "Result considered for the trial must be a float number: "
                "%s (%s)" % (str(result), str(type(result))))

        # return step_value, result
        return result

    def get_trials(self, query=None, projection=None, evaluations=False,
                   iterator=None):

        if iterator is None:
            iterator = iter

        query["experiment.name"] = {"$eq": self.name}

        # When not evaluating the test set, validate must always be True
        if not evaluations:
            query["config.validate"] = {"$eq": True}

        query.update(self.get_query_for_profile())

        rows = self.database.query(query, projection)
        for row in iterator(rows):
            if not self.space.validate(row["config"]):
                # raise RuntimeError("Invalid row %d" % row["_id"])
                logger.debug("Invalid row %d" % row["_id"])
                continue

            yield self._build_trial(row)

            # if evaluations or self.space.validate(row["config"]):
            #    yield self._build_trial(row)

    def get_query_for_profile(self):
        query_for_profile = dict()
        for profile_name, profile in self.space.iter_profiles():
            for hp_name, hp_value in profile.iteritems():
                query_for_profile["config.%s" % hp_name] = {"$eq": hp_value}

        return query_for_profile

    def get_runnable_trials(self, force_new=True):
        # Either the trial is queued or was interrupted on the same cluster
        trials = self.get_trials({
            "status": {
                "$in": status.RUNNABLE
            }}, {
            "config": 1, "status": 1})

        #     TODO: Find why this doesn't work.
        #     "$or": [
        #         {
        #             "status": {
        #                 "$in": status.QUEUED
        #             },
        #         },
        #         {
        #             "status": {
        #                 "$in": status.INTERRUPTED
        #             },
        #             "host.cluster_name": {
        #                 "$eq": CLUSTER_NAME
        #             }
        #         }
        #     ]})

        # Avoid the need of creating a list by "copying" the generator
        trials = (trial for trial in trials
                  if trial.id not in self.excluded_trials)

        trials, test_trials = itertools.tee(trials)

        if force_new and not any(True for i in test_trials):
            trials = self.create_new_trials()

        return trials

    def exclude(self, trial):
        self.excluded_trials.add(trial.id)

    def get_completed_trials(self, **kwargs):
        return self.get_trials({"status": {"$in": status.COMPLETED}}, **kwargs)

    def create_new_trials(self):
        logger.info("Creating new trials")

        strategy = self.optimizer.strategy

        x = []
        y = []
        for trial in self.get_trials({}):
            x.append(self.space.dict_to_list(trial.setting))

            if trial.status == "COMPLETED":
                y.append(trial.result[1])
            else:
                if strategy == "cl_min":
                    y_lie = numpy.min(y) if y else 0.0
                elif strategy == "cl_mean":
                    y_lie = numpy.mean(y) if y else 0.0
                else:
                    y_lie = numpy.max(y) if y else 0.0

                y.append(y_lie)

        x = self.optimizer.get_new_candidates(x, y)
        return self.register_settings(x)

    def register_settings(self, settings):

        # TODO might be better to do a direct count rather than iterating over
        # trials through the interface
        # (nevertheless there should not be many
        #  trials since there was not a single one a short time before this
        #  method was called)
        runnable_trials = list(self.get_runnable_trials(force_new=False))

        if len(runnable_trials) > 0:
            logger.info(
                "Some trials changed of status and became runnable during the "
                "sampling of new ones. The new ones are now discarded.")
            return runnable_trials

        trials = []
        for hp_list in settings:
            setting = self.space.list_to_dict(hp_list)
            trial = self._build_trial({'config': setting})
            trial.queue()
            trials.append(trial)

        return trials
