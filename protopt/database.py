import logging

from sacred.observers import MongoObserver

import smartdispatch.utils

import protopt.status
from protopt.utils import get_mongodb_url


CLUSTER_NAME = smartdispatch.utils.detect_cluster()

logger = logging.getLogger()


class Database(object):

    def __init__(self, name, collection, host_names, ports, user_name, password, ssl=False,
                 ssl_ca_file=None, replica_set=None, auth_source=None):

        self.name = name
        self.collection = collection
        self.host_names = host_names
        self.ports = ports
        self.user_name = user_name
        self.password = password
        self.ssl = ssl
        self.ssl_ca_file = ssl_ca_file
        self.replica_set = replica_set
        self.auth_source = auth_source

        self.mongo_observer = self._build_mongo_observer()
        self.runs = self.mongo_observer.runs
        self.metrics = self.mongo_observer.metrics
        self.fs = self.mongo_observer.fs

    def build_mongo_observer(self):
        logger.debug("Reusing database")
        return MongoObserver(runs_collection=self.mongo_observer.runs,
                             fs=self.mongo_observer.fs,
                             metrics_collection=self.mongo_observer.metrics,
                             overwrite=None)

    def _build_mongo_observer(self):
        logger.debug("Opening database %s with collection %s" %
                     (self.name, self.collection))
        options = {}
        if self.ssl:
            options["ssl"] = "true"
        if self.ssl_ca_file:
            options["ssl_ca_certs"] = self.ssl_ca_file
        if self.replica_set:
            options["replicaSet"] = self.replica_set
        if self.auth_source:
            options["authSource"] = self.auth_source

        mongo_url = get_mongodb_url(
            list(zip(self.host_names, self.ports)),
            user=self.user_name,
            password=self.password)
        # **options)

        # get_mongodb_url((("sacred-shard-00-00-85hpi.mongodb.net", 27017),
        # ("sacred-shard-00-01-85hpi.mongodb.net", 27017),
        # ("sacred-shard-00-02-85hpi.mongodb.net", 27017)), "bouthilx",
        # "02773469", "Sacred",

        # test_mongo_db(mongo_url, opt.name, table_name="runs",
        #               timeout=15, tries=60)
        mongodb_observer = MongoObserver.create(
            url=mongo_url, db_name=self.name,
            collection=self.collection, **options)

        return mongodb_observer

    def query(self, query=None, projection=None):
        if query is None:
            query = {}

        if projection is None:
            projection = {"config": 1, "result": 1, "status": 1, "metrics": 1}

        rows = self.runs.find(
            query, projection)

        return rows

    def find_job(self):
        logger.info("Looking for a new job")
        rows = self.runs.find({
            "status": {
                "$in": protopt.status.RUNNABLE
            },
            "$or": [
                {
                    "info.cluster": {
                        "$exists": False
                    }
                },
                {
                    "info.cluster": {
                        "$eq": CLUSTER_NAME
                    }
                }
            ]},
            {"id": 1, "config": 1})

        logger.debug("Counting jobs")
        count = rows.count()
        logger.info("%d jobs available" % count)
        # if count > 1:
        #    random_index = numpy.random.randint(count)
        if count >= 1:
            random_index = 0
        else:
            return None

        random_job = rows[random_index]
        selected_id = random_job["_id"]
        logger.debug("Selected id: %d (@index %d)" %
                     (selected_id, random_index))
        return random_job

    def select_random_config(self, space):
        row = self.find_job()
        if row is not None:
            config = row['config']
            space.force_options(config)
            return row['_id'], row, config
        else:
            return None, None, None
