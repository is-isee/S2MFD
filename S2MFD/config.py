import numpy as np
import json, os, sys
import importlib

class config_c:
    def __init__(self,parameter_file='parameters/defaults.py'):
        # Resolve the absolute path of the parameter file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        parameter_file = base_dir+'/'+parameter_file
                
        # Load the parameter file
        self.load_parameters(parameter_file)
        
    def load_parameters(self, parameter_file):
        """Load parameters from an external Python file."""
        # Load the parameter_file as a module
        spec = importlib.util.spec_from_file_location("parameters", parameter_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Set self attributes for all items in the module
        for k, v in vars(module).items():
            if not k.startswith("__"):  # Skip special attributes
                setattr(self, k, v) 
        
    def save(self):
        """Save configuration to a JSON file."""
        params = {}
        for k in dir(self):
            if not k.startswith("__") and not callable(getattr(self, k)):
                value = getattr(self, k)
                # JSONシリアライズ可能なデータ型のみ追加
                if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                    params[k] = value
        with open(self.datadir+self.configfile, 'w') as f:
            json.dump(params, f, indent=4)

    @classmethod
    def load(cls, filename):
        """Load configuration from a JSON file."""
        obj = cls.__new__(cls)
        with open(filename, 'r') as f:
            params = json.load(f)
        for k, v in params.items():
            setattr(obj, k, v)
            
        return obj