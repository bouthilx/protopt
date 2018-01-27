import copy
import os
import signal
import subprocess
from utils import TimeoutInterrupt


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

    # set gpu id in env var.
    gpu_id = args.pop("gpu_id")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    script = args.pop("script")

    # Build command line based on arguments
    arguments = " ".join("--%s %s" % (key, value)
                         for (key, value) in args.iteritems())

    signal.signal(signal.SIGTERM, sigterm_handler)

    # Execute the command
    command = "%s %s" % (script, arguments)
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)

    info = []
    output = 'non empty for starter'
    # Parse process' output
    while output != '' or process.pool() is not None:

        try:
            output = process.stdout.readline()
            if 'train_m' in output:
                info += output.strip().split('\t')[1:]
                info[0] = info[0].lstrip('#')
            if not output.startswith('#'):
                output = output.strip().split('\t')
                values = []
                try:  # TODO weak
                    values.append([float(x) for x in output])
                except ValueError:
                    continue
                assert info[0] == 'n'
                epoch = values[0]
                for key, value in zip(info[1:], values[1:]):
                    _run.log_scalar(key, float(value), epoch)
            # process output
            # fetch line
            # if line contains something important like train_m
            #     (https://github.com/Thrandis/my-pytorch-cifar/blob/master/plot.py#L68)
            # log it with the _run object.
        except KeyboardInterrupt as e:
            raise e

    rc = process.pool()

    # Catch some errors?

    # Catch KeyboardInterrupt and Timeout:
    #   # Kill the process, set status as interrupted.

    return rc
