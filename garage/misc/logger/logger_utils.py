import json
import os
from enum import Enum

from garage.misc.console import mkdir_p
from garage.misc.autoargs import get_all_parameters


def log_parameters_lite(log_file, args):
    log_params = {}
    for param_name, param_value in args.__dict__.items():
        log_params[param_name] = param_value
    if args.args_data is not None:
        log_params["json_args"] = dict()
    mkdir_p(os.path.dirname(log_file))
    with open(log_file, "w") as f:
        json.dump(log_params, f, indent=2, sort_keys=True, cls=MyEncoder)


def log_parameters(log_file, args, classes):
    log_params = {}
    for param_name, param_value in args.__dict__.items():
        if any([param_name.startswith(x) for x in list(classes.keys())]):
            continue
        log_params[param_name] = param_value
    for name, cls in classes.items():
        if isinstance(cls, type):
            params = get_all_parameters(cls, args)
            params["_name"] = getattr(args, name)
            log_params[name] = params
        else:
            log_params[name] = getattr(cls, "__kwargs", dict())
            log_params[name][
                "_name"] = cls.__module__ + "." + cls.__class__.__name__
    mkdir_p(os.path.dirname(log_file))
    with open(log_file, "w") as f:
        json.dump(log_params, f, indent=2, sort_keys=True)


def dump_variant(log_file, variant_data):
    mkdir_p(os.path.dirname(log_file))
    with open(log_file, "w") as f:
        json.dump(variant_data, f, indent=2, sort_keys=True, cls=MyEncoder)


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum':
                o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {'$function': o.__module__ + "." + o.__name__}
        return json.JSONEncoder.default(self, o)
