import logging
import os

import numpy

from sacred.commandline_options import CommandLineOption
from sacred.observers import MongoObserver
from sacred.utils import join_paths

import smartdispatch.utils

from protopt.utils import SacredSelectionError


logger = logging.getLogger(__name__)


STOPPED_STATES = ["QUEUED", "INTERRUPTED", "TIMED_OUT"]


class SelectOption(CommandLineOption):
    """ Select job with given id or {first, random, last}"""

    short_flag = "q"
    arg = "job_id"
    arg_description = "Job id in MongoDB to run"

    @classmethod
    def get_id(cls, args, table):
        if args in ["first", "last", "random"]:
            rows = table.find(
                {"status":
                    {"$in": STOPPED_STATES}})

            if args == "first":
                print(rows.count())
                return rows[0]["_id"]
            elif args == "last":
                return rows[rows.count() - 1]["_id"]
            else:  # args == "random":
                if rows.count() > 1:
                    random_index = numpy.random.randint(rows.count())
                else:
                    random_index = 0

                selected_id = rows[random_index]["_id"]
                logger.info("Selected id: %d (@index %d)" %
                            (selected_id, random_index))
                return selected_id
        else:
            return int(args)

    @classmethod
    def apply(cls, args, run):
        mongodb_observers = [o for o in run.observers if isinstance(o, MongoObserver)]
        assert len(mongodb_observers) == 1

        mongodb_observer = mongodb_observers[0]

        # table = db.runs
        table = mongodb_observer.runs
        job_id = SelectOption.get_id(args, table)
        row = table.find_one({"_id": job_id})

        if row["status"] not in STOPPED_STATES:
            raise SacredSelectionError(
                "Cannot run a job which has status "
                "\"%s\"" % str(row["status"]))

        # Make sure the job is resumed on the same cluster it ran on
        if "host" in row and row["status"] in ["INTERRUPTED", "TIMED_OUT"]:
            current_cluster_name = smartdispatch.utils.detect_cluster()
            ran_on_cluster = row["host"].get("cluster", None)
            if (ran_on_cluster is not None and
                    current_cluster_name != ran_on_cluster):
                raise SacredSelectionError(
                    "Job started and paused on a different cluster: %s" %
                    ran_on_cluster)
            elif (ran_on_cluster is not None and
                    current_cluster_name == ran_on_cluster):
                if not os.path.exists(run.config["save_path"]):
                    raise SacredSelectionError(
                        "File not available (%s). Maybe the experiment was "
                        "run on another cluster after all?" %
                        run.config["save_path"])

            run.config["resume"] = True
            row["config"]["resume"] = True

        # Set it to running quickly to avoid race conditions
        result = table.update_one(
            {
                "_id": int(row['_id']),
                "status": {"$eq": row["status"]}
            },
            {
                "$set": {"status": "RUNNING"}
            })

        if not result.acknowledged:
            raise SacredSelectionError("Update of the status failed.")
        elif result.modified_count != 1:
            fresh_row = table.find_one({"_id": int(row['_id'])})
            if fresh_row["status"] != row["status"]:
                raise SacredSelectionError(
                    "Race condition: the selected trial status was changed "
                    "from '%s' to '%s' by another worker." %
                    (row["status"], fresh_row["status"]))
            else:
                raise SacredSelectionError("Trial dissapeared from db... "
                                           "scary.")

        row["status"] = "RUNNING"
        mongodb_observer.overwrite = row
        mongodb_observer.run_entry = None

        # Force sources to be similar otherwise sacred will always complain.
        # TODO: Why isn't sacred able to compare correctly similar sources?
        #       See in MongoClient.started_event comparison giving raise to
        #       "Sources don't match".
        run.experiment_info["sources"] = row["experiment"]["sources"]

        # run.config = row["config"]
        if "info" in row:
            run.info = row["info"]


class EnforceNewOption(CommandLineOption):
    """
    Enforce running a new configuration unless the existing one is PAUSED,
    then resume it.
    """

    @classmethod
    def apply(cls, args, run):
        RTOL = 0.01

        mongodb_observers = [o for o in run.observers if isinstance(o, MongoObserver)]
        assert len(mongodb_observers) == 1

        mongodb_observer = mongodb_observers[0]

        # table = db.runs
        table = mongodb_observer.runs

        row = find_config(mongodb_observer, run.config, table, RTOL)

        if row:
            if row['status'] not in STOPPED_STATES:
                raise RuntimeError("Cannot run a job which has status "
                                   "\"%s\"" % str(row["status"]))

            run.config["resume"] = True
            row["config"]["resume"] = True

            # Set it to running quickly to avoid race conditions
            table.update({"_id": int(row['_id'])},
                         {"$set": {"status": "RUNNING"}})

            row["status"] = "RUNNING"
            mongodb_observer.overwrite = row

        else:
            command = join_paths(run.main_function.prefix,
                                 run.main_function.signature.name)

            run.run_logger.info("Reserving setting in DB")
            mongodb_observer.started_event(
                ex_info=run.experiment_info,
                command=command,
                host_info=run.host_info,
                start_time=run.start_time,
                config=run.config,
                meta_info=run.meta_info,
                _id=None)

            # Save it in overwrite to keep it when run get's started for real
            row = mongodb_observer.overwrite = mongodb_observer.run_entry
            mongodb_observer.run_entry = None

        # Force sources to be similar otherwise sacred will always
        # complain.
        # TODO: Why isn't sacred able to compare correctly
        # similar sources?  See in MongoClient.started_event comparison
        # giving raise to "Sources don't match".
        run.experiment_info["sources"] = row["experiment"]["sources"]

        # run.config = row["config"]
        run.info = row["info"]


def find_config(mongodb_observer, config, table, rtol):
    query = create_comparison_query(config, rtol)
    rows = table.find(query)
    if rows.count() > 0:
        return rows[0]
    else:
        return None


def create_comparison_query(config, rtol=0.01):
    ignore = ["seed", "dataroot", "nthread", "resume", "save", "tensorboard",
              "verbose"]
    query = {}
    for hp_name, hp_value in config.items():
        if hp_name in ignore:
            continue

        if isinstance(hp_value, float):
            hp_comparison = {
                "$gte": hp_value * (1 - rtol),
                "$lte": hp_value * (1 + rtol)
            }
        else:
            # Use pure equality
            hp_comparison = {
                "$eq": hp_value
            }

        db_hp_name = "config.%s" % hp_name
        logger.debug("%s comparison interval: %s" %
                     (db_hp_name, str(hp_comparison)))

        query[db_hp_name] = hp_comparison

    return query
