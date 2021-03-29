# import os, glob
# from importlib import import_module
# from pathlib import Path

# modules = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))

# __all__ = [
#     os.path.basename(f)[:-3] for f in modules if not f.endswith("__init__.py")
# ]

# for each in __all__:
#     import_module(f".{each}", __package__)
