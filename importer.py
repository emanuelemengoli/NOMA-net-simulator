import os
import importlib.util
import sys

def import_all_modules(directory, recursive=True):
    for root, dirs, files in os.walk(directory):
        if not recursive:
            dirs[:] = []
        for file in files:
            if file.endswith('.py') and not file.startswith('__init__'):
                file_path = os.path.join(root, file)
                module_name = os.path.splitext(os.path.relpath(file_path, directory))[0].replace(os.path.sep, '.')
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                print(f"Imported {module_name}")