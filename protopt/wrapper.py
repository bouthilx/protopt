import copy
import logging
import os
import signal
import subprocess
from utils import TimeoutInterrupt


logger = logging.getLogger(__name__)


def sigterm_handler(signal, frame):
    if sigterm_handler.triggered:
        return
    else:
        sigterm_handler.triggered = True

    raise TimeoutInterrupt("Experiment killed by the scheduler")


sigterm_handler.triggered = False


def fake_fct_signature(defaults, _run, *args, **kwargs):
    assert len(args) == 0
    # args = dict_to_options(_run.config, impn.main.parse_args([]))
    # return impn.main.train(args, run=_run)

    args = copy.copy(defaults)
    args.update(_run.config)

    import pprint
    pprint.pprint(args)

    # set gpu id in env var.
    gpu_id = args.pop("gpu_id")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    script = args.pop("script")

    #
    args.pop("validate")
    args.pop("seed")

    # Build command line based on arguments
    arguments = ""
    for (key, value) in args.iteritems():
        if isinstance(value, bool) and value:
            arguments += "--%s " % key
        elif not isinstance(value, bool):
            arguments += "--%s %s " % (key, value)
    arguments = arguments.strip()

    signal.signal(signal.SIGTERM, sigterm_handler)

    # Execute the command
    command = "%s %s" % (script, arguments)
    logger.info("Running command:\n%s" % command)
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)

    info = ['n']
    output = 'non empty for starter'
    # Parse process' output
    while output != '' or process.poll() is None:

        try:
            logger.debug("Waiting for script's stdout")
            output = process.stderr.readline().lstrip("INFO:root:").strip()
            logger.info("stderr: %s" % output)
            if 'train_m' in output:
                assert output.strip().split('\t')[0].lstrip('#') == 'n'
                info += output.strip().split('\t')[1:]
                logger.info("info: %s" % str(info))
            if not output.startswith('#'):
                columns = output.strip().split('\t')
                try:  # TODO weak
                    values = [float(x) for x in columns]
                except ValueError:
                    logger.info("continue")
                    continue
                epoch = values[0]
                for key, value in zip(info[1:], values[1:]):
                    logger.info("Logging %s: (%s, %s)" %
                                (key, str(epoch), str(value)))
                    _run.log_scalar(key, float(value), epoch)
            # process output
            # fetch line
            # if line contains something important like train_m
            #     (https://github.com/Thrandis/my-pytorch-cifar/blob/master/plot.py#L68)
            # log it with the _run object.
        except KeyboardInterrupt as e:
            raise e

    rc = process.poll()

    if rc > 0:
        raise RuntimeError(process.stderr.read())

    # Catch some errors?

    # Catch KeyboardInterrupt and Timeout:
    #   # Kill the process, set status as interrupted.

    return rc
