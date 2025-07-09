from importlib import import_module
from typing import Any


class DotDict(dict):
    """Dictionary with attribute-style access.

    Example
    -------
    >>> cfg = DotDict({'a': 1, 'b': {'c': 2}})
    >>> cfg.a
    1
    >>> cfg.b.c
    2
    """

    def __getattr__(self, item: str) -> Any:  # noqa: D401
        try:
            value = self[item]
            # Automatically convert nested dicts/lists to DotDict where relevant
            if isinstance(value, dict) and not isinstance(value, DotDict):
                value = DotDict(value)
                self[item] = value
            elif isinstance(value, list):
                value = [DotDict(v) if isinstance(v, dict) else v for v in value]
                self[item] = value
            return value
        except KeyError as exc:
            raise AttributeError(item) from exc

    # Attribute-style assignment/deletion delegate to dict methods
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def copy(self):  # noqa: D401
        return DotDict({k: v for k, v in self.items()})


def _convert_to_dotdict(obj: Any) -> Any:
    """Recursively convert dicts/lists into DotDict instances."""
    if isinstance(obj, dict) and not isinstance(obj, DotDict):
        return DotDict({k: _convert_to_dotdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_convert_to_dotdict(v) for v in obj]
    return obj


def instantiate_from_target(cfg: Any):
    """Instantiate an object from a Hydra-style ``_target_`` dictionary.

    This is a minimal replacement for ``hydra.utils.instantiate`` that supports:
    - Nested dictionaries/lists (recursively instantiated)
    - Returning already-constructed objects unchanged
    - Dataclass/objects providing a ``build`` method (called eagerly)
    """
    # If the object already knows how to build itself, do it immediately.
    if hasattr(cfg, "build") and callable(getattr(cfg, "build")):
        return cfg.build()

    # Lists/tuples → recurse
    if isinstance(cfg, list):
        return [instantiate_from_target(c) for c in cfg]
    if isinstance(cfg, tuple):
        return tuple(instantiate_from_target(c) for c in cfg)

    # Non-config objects are returned as-is
    if not isinstance(cfg, dict):
        return cfg

    cfg_dict = dict(cfg)  # shallow copy so we can pop safely
    target = cfg_dict.pop("_target_", None)
    if target is None:
        # Not a target dict; attempt recursive conversion but return unchanged
        return _convert_to_dotdict(cfg_dict)

    module_path, cls_name = target.rsplit(".", 1)
    module = import_module(module_path)
    cls = getattr(module, cls_name)

    # Recursively instantiate nested structures
    params = {k: instantiate_from_target(v) for k, v in cfg_dict.items()}

    return cls(**params) 