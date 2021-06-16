# -*- coding: utf-8 -*-

__all__ = [
    'BaseClass'
]


###########
# CLASSES #
###########

class BaseClass(type):

    def __new__(mcs, name, bases, classes):

        for b in bases:
            if isinstance(b, BaseClass):
                raise TypeError(f"Type '{b.__name__}' is not an acceptable base type.")

        return type.__new__(mcs, name, bases, dict(classes))
