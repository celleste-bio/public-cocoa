import importlib.util
import os
import sys

def go_back_dir(path, number):
    if os.path.isfile(path):
        path = os.path.dirname(path)
    
    result = path
    for i in range(number):
        result = os.path.dirname(result)
    return result

def import_from(file_path, function_name):
    absolute_path = os.path.abspath(file_path)
    module_name = os.path.splitext(os.path.basename(absolute_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, absolute_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    func = getattr(module, function_name)
    return func
