# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import hta
from hta.configs.default_values import DEFAULT_CONFIG_FILENAME

ConfigValue = Union[None, bool, int, float, str, Dict[str, Any], List[Any], Set[Any]]


def setup_logger(config_file: str = "$HTA/configs/logging.config") -> logging.Logger:
    global logger
    if config_file:
        if config_file.startswith("$HTA"):
            config_file = str(Path(hta.__file__).parent.joinpath(config_file[5:]))
        logging.config.fileConfig(config_file)
        logger = logging.getLogger("hta")
    elif logger is None:
        logger = logging.getLogger()
    return logger


logger: logging.Logger = setup_logger()


class HtaConfig:
    """
    A container for customizing the trace analyzer configurations.
    """

    @staticmethod
    def get_default_paths() -> List[str]:
        """
        Get the list of default config file paths.

        Returns:
            A list of default config file paths.

        Notes:
        To support the customization of trace analysis, hta reads configurations from
            a list of config file paths in the following order:
            1. hta/configs/trace_analyzer.json
            2. ~/.hta/trace_analyzer.json
            3. <current directory>/trace_analyzer.json
            4. a user specified, instance-specific configuration file path.

        Loading configurations from the subsequent configuration file will update previous configurations.

        If a configuration file in the above list doesn't exist, hta will skip such configuration file.
        As a class method, this function does not return the user defined config file path.
        """
        return [
            str(Path(hta.__file__).parent.joinpath("configs").joinpath(DEFAULT_CONFIG_FILENAME)),
            str(Path.home().joinpath(".hta").joinpath(DEFAULT_CONFIG_FILENAME)),
            str(Path.cwd().joinpath(DEFAULT_CONFIG_FILENAME)),
        ]

    def __init__(
        self,
        config_file_path: Optional[str] = None,
        load_default_paths: Optional[bool] = True,
    ):
        """
        Constructor of the HtaConfig class.

        Args:
             config_file_path (str): a user provided config file path.
             load_default_paths (bool) : control whether the analyzer should use available default configuration files.
        """
        self.config_file_paths: List[str] = HtaConfig.get_default_paths() if load_default_paths else []
        if config_file_path:
            self.config_file_paths.append(config_file_path)

        self.config: Dict[str, Any] = {}
        for path in self.config_file_paths:
            if Path(path).exists() and os.access(path, os.R_OK):
                with open(path) as fp:
                    cfg = json.load(fp)
                    self.config.update(cfg)

        logging_config_file = self.get_config("hta.defaults.logging.config_file")
        if isinstance(logging_config_file, str):
            setup_logger(logging_config_file)
        else:
            setup_logger()

    def get_config_file_paths(self) -> List[str]:
        """
        Return the available config file paths
        """
        return [path for path in self.config_file_paths if Path(path).exists() and os.access(path, os.R_OK)]

    def get_config(self, dot_path: Optional[str] = None, default_value: ConfigValue = None) -> ConfigValue:
        """
         Return the value for a configuration key <doct_keys>.

        Args:
             dot_path (str) : a dot representation of a configuration path. For example, `a.b.c` will search the
             configuration for self.configs['a']['b']['c']. If dot_path is None, then the method will return
                 the entire set of configurations.
             default_value (ConfigValue) : return value if the dot_path doesn't exist in the configuration.

         Returns:
             The configuration value corresponding to the dot_path.

         Notes:
             HtaConfig represents its configuration using a dictionary which can have multiple layers.
             For easy use, this method allows the user to specify the search path with a dot string and then splits
             the dot string into a list of keys to navigate the configuration search.
        """
        cfg = self.config
        if dot_path is None:
            return self.config

        paths = dot_path.split(".")
        found = False
        for i, path in enumerate(paths):
            if cfg is None:
                break
            if isinstance(cfg, Dict) and path in cfg:
                cfg = cfg[path]
                if i == len(paths) - 1:
                    found = True
            else:
                break

        if found:
            return cfg
        else:
            return default_value

    def show(self):
        print(json.dumps(self.config, indent=4, sort_keys=True))
