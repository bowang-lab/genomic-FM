import os
from os import PathLike
from typing import Union


PathOrStr = Union[str, PathLike]
def get_bytes_range(source, bytes_start: int, num_bytes: int) -> bytes:
    with open(source, "rb") as f:
        f.seek(bytes_start)
        return f.read(num_bytes)

def file_size(path) -> int:
    """
    Get the size of a local or remote file in bytes.
    """
    return os.stat(path).st_size
