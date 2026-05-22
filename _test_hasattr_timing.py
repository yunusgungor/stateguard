"""Minimal test to verify __init_subclass__ timing with ABCMeta."""
from abc import ABC, ABCMeta, abstractmethod


class WatcherMeta(ABCMeta):
    """ABCMeta subclass that traces attribute state."""

    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        print(f"[WatcherMeta.__new__] {name}: __abstractmethods__ = {cls.__abstractmethods__}")
        print(f"[WatcherMeta.__new__] {name}: hasattr = {hasattr(cls, '__abstractmethods__')}")
        return cls


class BaseValidator(ABC, metaclass=WatcherMeta):
    """Simulates StateGuard's BaseValidator."""

    name: str = "base"
    dimension: str = "structural"
    tier: int = 1

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        print(f"[__init_subclass__] {cls.__name__}:")
        print(f"  ABC in bases: {ABC in cls.__bases__}")
        own_am = cls.__dict__.get("__abstractmethods__", "NOT_IN_OWN_DICT")
        print(f"  __abstractmethods__ in cls.__dict__: {own_am}")
        print(f"  hasattr(__abstractmethods__): {hasattr(cls, '__abstractmethods__')}")
        if hasattr(cls, "__abstractmethods__"):
            print(f"  value: {cls.__abstractmethods__}")
            print(f"  bool: {bool(cls.__abstractmethods__)}")

    @abstractmethod
    def validate(self, output, context=None):
        ...


class ConcreteValidator(BaseValidator):
    name = "concrete"
    dimension = "quantitative"
    tier = 2

    def validate(self, output, context=None):
        return "OK"


print("\n=== Creating AbstractMid ===")

class AbstractMid(BaseValidator):
    @abstractmethod
    def validate(self, output, context=None):
        ...


print("\n=== Creating ConcreteChild ===")

class ConcreteChild(AbstractMid):
    name = "child"
    dimension = "semantic"
    tier = 3

    def validate(self, output, context=None):
        return "OK"


print("\n=== Tests ===")
v = ConcreteChild()
print(f"ConcreteChild works: {v.validate('test')}")
print("All tests passed!")
