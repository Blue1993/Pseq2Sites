import yaml
import pprint
import os
import logging
import sys

import pandas as pd
import csv

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.DEBUG,
    stream=sys.stdout,
)

def load_cfg(yaml_filepath):
    
    ### Load a yaml configuration file.
    """
    # Parameters
        - yaml file path: str
    # Returns
        - cfg: dict
    """
    
    with open(yaml_filepath, "r") as f:
        cfg = yaml.safe_load(f)
        
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    
    return cfg
    
    #return Config(cfg)

def make_paths_absolute(dir_, cfg):
    
    ### Make all values for keys ending with '_path' absolute to dir_.
    
    """
    # Parameters
        - dir_: str
        - cfg: dict
    # Returns
        - cfg: dict
    """
    
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.isfile(cfg[key]) and not os.path.isdir(cfg[key]):
                logging.error("%s does not exist.", cfg[key])

        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    
    return cfg

