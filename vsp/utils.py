# -*- coding: utf-8 -*-
"""Utility classes and functions.
"""

import functools


def strip_none_args(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = [a for a in args if a is not None]      
        result = func(*args, **kwargs) if args else func(**kwargs)
        return result
    return wrapper


#def compose(*funcs):
#    return functools.reduce(lambda f, g: lambda *args, **kwargs: \
#                            f(g(*args, **kwargs)), funcs)


#def compose(*funcs):
#    def inner(*args, **kwargs):
#        for f in reversed(funcs):
#            result = f(*args, **kwargs)
#            args = (result, )
#        return result
#    return inner


class CompositeFunction:
    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, *args, **kwargs):
        for f in reversed(self.funcs):
            result = f(*args, **kwargs)
            args = (result, )
        return result


def compose(*funcs):
    return CompositeFunction(*funcs)
