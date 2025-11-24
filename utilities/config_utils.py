import argparse
import enum

import ml_collections
from termcolor import colored


def flatten_config_dict(config: ml_collections.ConfigDict):
    fields = []
    for k in config:
        v = config[k]
        if isinstance(v, ml_collections.ConfigDict):
            for sub_k, sub_v in flatten_config_dict(v):
                fields.append((f"{k}.{sub_k}", sub_v))
        else:
            fields.append((k, v))
    return fields


def parse_cli_overides(config: ml_collections.ConfigDict):
    """
    Parse args from CLI and override config dictionary entries
    """
    parser = argparse.ArgumentParser()
    flatten_config = flatten_config_dict(config)
    for key, value in flatten_config:
        parser.add_argument(f"--{key}")
    args = vars(parser.parse_args())

    def print_config_override(key, old_value, new_value, first_config_overide):
        if first_config_overide:
            print(colored("Config overrides:", "red"))
        print(f"     {key:25s} -> {new_value} (instead of {old_value})")

    def cast_argument(old_value, new_value):
        try:
            if new_value is None:
                return None
            if type(old_value) is int:
                return int(new_value)
            if type(old_value) is float:
                return float(new_value)
            if type(old_value) is str:
                return new_value
            if type(old_value) is bool:
                return new_value.lower() in ("yes", "true", "t", "1")
            if type(old_value) in [tuple, list]:
                sequence_constructor = type(old_value)
                old_element = old_value[0]
                return sequence_constructor(
                    cast_argument(old_element, e) for e in new_value.split(",")
                )
            if issubclass(old_value.__class__, enum.Enum):
                return old_value.__class__(new_value)
            if old_value is None:
                return new_value  # assume string
            raise ValueError()
        except Exception:
            raise ValueError(f"Unable to parse config key '{key}' with value '{new_value}'")

    first_config_overide = True
    for key, original_value in flatten_config:
        override_value = cast_argument(original_value, args[key])
        if override_value is not None and override_value != original_value:
            c = config
            for k in key.split(".")[:-1]:
                c = c[k]
            c[key.split(".")[-1]] = override_value
            # setattr(config, key, override_value)
            print_config_override(key, original_value, override_value, first_config_overide)
            first_config_overide = False

    return config
