
import sys, os
import io
from inspect import getframeinfo, stack, currentframe
import inspect
from typing import overload
from glob import glob
'''

import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
__dir__ = os.path.abspath( os.path.join(__dir__, '..') )
if __dir__ not in sys.path: sys.path.append(__dir__)
os.chdir(__dir__)

''';

def static_vars(**kwargs):
    '''
    @static_vars(sep=' ')
    '''
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

class DictToObj:
    def __init__(self, adict):
        assert isinstance(adict, dict)
        self.__dict__.update(adict)

def unique_name(name, use_time=True):
    import datetime
    if use_time:
        # date = datetime.datetime.now().strftime("_%Y%m%d_%H%M") # YYYYMMDD_HHmm
        date = datetime.datetime.now().strftime("_%Y%m%d") # YYYYMMDD
        name += date
    return name


def reload(module_name):
    '''
    reload('sys')
    '''
    if module_name in sys.modules:
        del sys.modules[module_name]


import signal
@static_vars(sigstate=0)
def is_sigint():
    if is_sigint.sigstate == 0:
        def handler(sig, frame): is_sigint.sigstate = 2
        signal.signal(signal.SIGINT, handler)
        is_sigint.sigstate = 1
    return is_sigint.sigstate == 2



def dmsg(*args, **kwargs):
    caller = inspect.stack()[1][3]
    callerline = getframeinfo(currentframe().f_back.f_back).lineno
    print(f'[{callerline}] ', end='')
    print(*args, **dict(kwargs, sep=" ____ ")) # sep for more notiable output.

#
# juypter helper
#


def stop_here(msg='Early stopped!'):
    '''
    stop execute a cell without reset kernel.
    '''
    import inspect

    try:
        class EarlyStop(InterruptedError):
            pass

        def except_handler(shell, etype, evalue, tb, tb_offset=None):
            sys.stderr.close()
            sys.stderr = sys.__stderr__

        get_ipython().set_custom_exc((EarlyStop,), except_handler)
        sys.stderr = io.StringIO()
    except NameError as ex:
        err(ex)
        return

    lineno = inspect.getframeinfo(inspect.currentframe().f_back).lineno
    print(f'\x1b[33;40m\n      {msg} at line {lineno}     \n\x1b[0m')
    raise EarlyStop



class ScopedInstance:
    def __init__(self, instance):
        super().__init__()
        self.instance = instance

    def __enter__(self):
        return self.instance

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.instance:
            del self.instance


def print_codes_only_be_used_in_ipynb():
    print('''

nb_folder = globals()['_dh'][0] # get this local notebook(xxx.ipynb) folder
    ''')


def caller_test():
    caller = inspect.stack()[1] # frame, filename, lineno, function
    print(caller.filename, caller.lineno, caller.function)


@static_vars(sep=' ')
def dprint(*args, **kwargs):
    caller = inspect.stack()[1] # frame, filename, lineno, function
    # print(caller.filename, caller.lineno, caller.function)

    kwargs['sep'] = dprint.sep

    header = f'{caller.function}() {caller.lineno}'
    __builtins__.print(f'{header:<30}', end=': ') # func.line with fixed length
    __builtins__.print(*args, **kwargs) # sep for more notiable output.


def error(*args, **kwargs):
    RED, YELLOW, GREEN, PINK, CYAN= 91, 40, 92, 95, 36
    caller = inspect.stack()[1] # frame, filename, lineno, function

    # color: https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
    header = f'{caller.function}() {caller.lineno}'
    __builtins__.print(f'\x1b[33;{RED}m{header:<30}', end=': ') # func.line with fixed length

    kwargs['sep'] = dprint.sep
    kwargs['end'] = '\x1b[0m\n'
    __builtins__.print(*args, **kwargs)


#
#
#
class DotDict(dict):
    """A dictionary that's recursively navigable with dots, not brackets.
        cfg = DotDict(dict(abc=123, def='abc'))
        print(cfg.abc, cfg,def)
    """
    def __init__(self, data: dict = None):
        super().__init__()
        assert isinstance(data, dict), f"{type(self) is not dict}"
        for key, value in data.items():
            if isinstance(value, list):
                self[key] = [DotDict(item) for item in value]
            elif isinstance(value, dict):
                self[key] = DotDict(value)
            else:
                self[key] = value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"attribute .{key} not found")


def get_latest_files(folder_path, n_count=1):
    files = [i for i in glob(folder_path + '/*') if os.path.isfile(i)]
    sorted_files = sorted(files, key=lambda x: os.path.getmtime(x), reverse=True)
    return sorted_files[:n_count]
#
#
#

def txtclr(code = 0):
    ''' 0: reset, <0: print all '''
    if not hasattr(txtclr, "CLRS"):
        txtclr.CLRS = [f'\33[0m', *[f'\33[{i}m' for i in range(30,38)]]

    if 0<= code and code < len(txtclr.CLRS):
        print(txtclr.CLRS[code], end='')
    else:
        for i in range(len(txtclr.CLRS)):
            print(i,':', txtclr.CLRS[i] + 'txtclr test...', txtclr.CLRS[0])



def unittest():
    print('test...', 'b')
    error(1,2,3)



if __name__ == '__main__':
    dprint.sep = ' ~~ '
    print = dprint

    unittest()