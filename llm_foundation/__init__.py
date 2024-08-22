"""
llm_foundation module
"""
import logging
import os
from logging.config import fileConfig
from typing import List

import coloredlogs
from dotenv import load_dotenv, find_dotenv

# import sys
# import pkg_resources  # type: ignore
# if sys.version_info >= (3, 8):
#     from importlib import metadata
# else:
#     import importlib_metadata as metadata
# __version__: str = metadata.version(__name__) #"0.0.1"  # pkg_resources.get_distribution("llm_base").version


load_dotenv(find_dotenv()) # read local .env file and put it in os.environ

__all__: List[str] = []
__copyright__: str = "Copyright 2024, Francisco Perez-Sorrosal."

env_key = "LOGGER"
logger_name = os.getenv(env_key, "root")

env_key = "LOG_CFG"
config_path = os.getenv(env_key, None)

env_key = "LOG_DEST"
home_dir = os.getenv("HOME", "/var/log")
log_dir = os.getenv(env_key, home_dir)
env_key = "LOG_DEST_FILE"
log_filaname = os.getenv(env_key, "latte.log")
logfilename_path = os.path.join(log_dir, log_filaname)

env_key = "LOG_LEVEL"
log_level = os.getenv(env_key, "INFO")

print(f"Trying to configure logger {logger_name} in module {__name__}")
logger = logging.getLogger(logger_name)
print(f"{logger.name} # of associated handlers - {len(logger.handlers)}")

if len(logging.getLogger(logger_name).handlers) <= 0:
    print("Logging is not configured yet. Configuring it now.")

    if config_path is None or not os.path.exists(config_path):
        print("Basic logging config")
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
    else:
        print(f"Loading logging config from file {config_path}. Output to: {logfilename_path}")
        if not os.path.exists(log_dir):
            print(f"Creating dir {log_dir} for placing the log as it does not exist yet")
            os.makedirs(log_dir)
        fileConfig(
            config_path,
            defaults={"logfilename": logfilename_path, "loglevel": log_level},
            disable_existing_loggers=False,
        )
        logger = logging.getLogger(logger_name)
    coloredlogs.install(logger=logger)
    logger.info(f"Logger {logger_name} configured")
else:
    logger.info(f"Logger {logger_name} already configured")
