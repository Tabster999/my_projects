"""Auto-export public functions from all module files in this package.

Usage
-----
	import modules as myf
	# then use myf.some_function(...)

Rules
-----
- Every `.py` file in this folder (except private files starting with `_`) is imported.
- Every public function (name not starting with `_`) defined in those files is re-exported.
"""

from importlib import import_module
import inspect
import pkgutil

_exports: list[str] = []

for _, _mod_name, _is_pkg in pkgutil.iter_modules(__path__):
	if _is_pkg or _mod_name.startswith("_"):
		continue

	_mod = import_module(f"{__name__}.{_mod_name}")
	globals()[_mod_name] = _mod
	_exports.append(_mod_name)

	for _name, _obj in vars(_mod).items():
		if _name.startswith("_"):
			continue
		if inspect.isfunction(_obj) and _obj.__module__ == _mod.__name__:
			globals()[_name] = _obj
			_exports.append(_name)

# keep exports stable and unique
__all__ = sorted(set(_exports))  # pyright: ignore[reportUnsupportedDunderAll]

