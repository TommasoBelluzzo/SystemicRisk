# -*- coding: utf-8 -*-

__all__ = [
    'alias',
    'cachedproperty',
    'aliased'
]


###########
# IMPORTS #
###########

# Partial

from functools import (
    update_wrapper,
    wraps
)

from threading import (
    RLock
)


###########
# CLASSES #
###########

# noinspection PyPep8Naming
class alias(object):

    """
    A decorator for implementing method aliases.

    | It can be used only inside @aliased-decorated classes.
    """

    def __init__(self, *aliases):

        self.aliases = set(aliases)

    def __call__(self, obj):

        if isinstance(obj, property):
            obj.fget._aliases = self.aliases
        else:
            obj._aliases = self.aliases

        return obj


# noinspection PyPep8Naming, PyUnusedLocal
class cachedproperty(property):

    """
    A decorator for implementing lazy-evaluated read-only properties.
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):

        doc = doc or fget.__doc__
        super(cachedproperty, self).__init__(fget, None, None, doc)

        self._func = fget
        self._func_name = None

        update_wrapper(self, fget)

        self._lock = RLock()

    def __set_name__(self, owner, name):

        if self._func_name is None:
            self._func_name = name
        elif name != self._func_name:
            raise AttributeError(f'Cannot assign the same cached property to two different names: {self._func_name} and {name}.')

    def __get__(self, instance, owner=None):

        if instance is None:
            return self

        if self._func_name is None:
            raise AttributeError('Cannot use a cached property without calling "__set_name__" on it.')

        with self._lock:
            try:
                return instance.__dict__[self._func_name]
            except KeyError:
                return instance.__dict__.setdefault(self._func_name, self._func(instance))

    def __set__(self, obj, value):

        if obj is None:
            raise AttributeError('The parameter "obj" is null.')

        raise AttributeError('This property cannot be set.')

    def deleter(self, fdel):

        raise AttributeError('This property cannot implement a deleter.')

    def getter(self, fget):

        return type(self)(fget, None, None, None)

    def setter(self, fset):

        raise AttributeError('This property cannot implement a setter.')


#############
# FUNCTIONS #
#############

# noinspection PyProtectedMember
def aliased(aliased_class):

    """
    A decorator for enabling aliases.
    """

    def wrapper(func):

        @wraps(func)
        def inner(self, *args, **kwargs):

            return func(self, *args, **kwargs)

        return inner

    aliased_class_dict = aliased_class.__dict__.copy()
    aliased_class_set = set(aliased_class_dict)

    for name, method in aliased_class_dict.items():

        if isinstance(method, property) and hasattr(method.fget, '_aliases'):
            aliases = method.fget._aliases
        elif hasattr(method, '_aliases'):
            aliases = method._aliases
        else:
            aliases = None

        if aliases is not None:
            for a in aliases - aliased_class_set:

                doc = method.__doc__
                doc_alias = doc[:len(doc) - len(doc.lstrip())] + 'Alias of **' + name + '**.'

                if isinstance(method, property):
                    wrapped_method = property(method.fget, method.fset, method.fdel, doc_alias)
                else:
                    wrapped_method = wrapper(method)
                    wrapped_method.__doc__ = doc_alias

                setattr(aliased_class, a, wrapped_method)

    return aliased_class
