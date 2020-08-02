from typing import Any, List, Dict


class Singleton(type):
    _instances: Dict[Any, Any] = {}

    def __call__(cls, *args: List[Any], **kwargs: Dict[str, Any]):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
