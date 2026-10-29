"""Local NumPy 1.x compatibility for PoreSpy 3.0.4 snow2's reshape keyword.

Only the copied snow2 function sees this proxy. No installed module, global
NumPy function or scientific parameter is modified.
"""
import inspect
import types
import functools
import numpy as np

class _NumpyProxy:
    def __getattr__(self,name): return getattr(np,name)

    @staticmethod
    def reshape(a,*args,**kwargs):
        if 'shape' in kwargs:
            if args or 'newshape' in kwargs: raise TypeError('Multiple reshape dimensions supplied')
            return np.reshape(a,kwargs.pop('shape'),**kwargs)
        return np.reshape(a,*args,**kwargs)

def compatible_snow2(original):
    if 'shape' in inspect.signature(np.reshape).parameters:
        return original
    copied_globals=dict(original.__globals__)
    copied_globals['np']=_NumpyProxy()
    copied=types.FunctionType(original.__code__,copied_globals,original.__name__,
                              original.__defaults__,original.__closure__)
    copied.__kwdefaults__=original.__kwdefaults__
    return functools.update_wrapper(copied,original)
