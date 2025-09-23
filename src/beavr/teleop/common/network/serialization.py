import pickle
from abc import ABC, abstractmethod
from typing import Generic, Optional, Type, TypeVar

T = TypeVar("T")


class Serializer(ABC, Generic[T]):
    """Abstract serializer interface for message payloads.

    Implementations must be symmetrical: ``decode(encode(x)) == x`` for supported inputs.
    """

    @abstractmethod
    def encode(self, obj: T) -> bytes:
        pass

    @abstractmethod
    def decode(self, buffer: bytes) -> T:
        pass


class PickleSerializer(Serializer[T]):
    """Pickle-based serializer with optional runtime type validation."""

    def __init__(self, expected_type: Optional[Type[T]] = None, allow_subclasses: bool = True):
        self._expected_type = expected_type
        self._allow_subclasses = allow_subclasses

    def encode(self, obj: T) -> bytes:
        return pickle.dumps(obj, protocol=-1)

    def decode(self, buffer: bytes) -> T:
        obj = pickle.loads(buffer)
        if self._expected_type is not None:
            if self._allow_subclasses:
                if not isinstance(obj, self._expected_type):
                    raise TypeError(f"Decoded type {type(obj)} != expected {self._expected_type}")
            else:
                if type(obj) is not self._expected_type:
                    raise TypeError(f"Decoded exact type {type(obj)} != expected {self._expected_type}")
        return obj


class RawBytesSerializer(Serializer[bytes]):
    """No-op serializer for raw byte streams."""

    def encode(self, obj: bytes) -> bytes:
        return obj

    def decode(self, buffer: bytes) -> bytes:
        return buffer
