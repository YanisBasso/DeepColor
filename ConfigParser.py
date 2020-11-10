#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:11:02 2020

@author: yanis
"""

from pathlib import Path
from datetime import datetime 
import json 
from collections import OrderedDict

class ConfigParser:
    
    def __init__(self,config,resume = None):
        
        self._config = config
        self.resume = resume 
        
        #Set save directory 
        save_dir = Path(config['trainer']['save_dir'])
        experience_name = self.config['name']
        run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / experience_name  / run_id
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._setup_save_dir()
        
        
    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.
        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)
    
    @classmethod
    def from_args(cls,args):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
            
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
            
        #Read JSON 
        with cfg_fname.open('rt') as handle:
            config = json.load(handle, object_hook=OrderedDict)
            
        return cls(config, resume, modification)
    
    def _setup_save_dir(self):
        self.save_dir.mkdir(parents=True,exist_ok=exist_ok)
        #Add config file to the directory 
        fname = self.save_dir / 'config.json'
        with fname.open('wt') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=False)
        
    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]
    
    @property 
    def config(self):
        return self._config
    
    @property 
    def save_dir(self):
        return self._save_dir
    
    