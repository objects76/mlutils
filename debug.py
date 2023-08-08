
# debug helpers
#
from inspect import getframeinfo, stack, currentframe

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

curline = lambda : getframeinfo(currentframe().f_back).lineno
callerline = lambda : getframeinfo(currentframe().f_back.f_back).lineno



def stop_here(msg='Early stopped!'):
    '''
    stop execute a cell without reset kernel.
    '''
    import inspect, io

    try:
        class EarlyStop(InterruptedError):
            pass

        def except_handler(shell, etype, evalue, tb, tb_offset=None):
            sys.stderr.close()
            sys.stderr = sys.__stderr__

        get_ipython().set_custom_exc((EarlyStop,), except_handler)
        sys.stderr = io.StringIO()
    except NameError as ex:
        print(ex)
        return

    lineno = inspect.getframeinfo(inspect.currentframe().f_back).lineno
    print(f'\x1b[33;40m\n      {msg} at line {lineno}     \n\x1b[0m')
    raise EarlyStop


def clrcode(n): return '\033['+str(n)+'m'
MARGENTA,BLUE,GREEN,GREY,YELLOW,RED,RESET = clrcode(35),clrcode(36),clrcode(92),clrcode(38),clrcode(33),clrcode(31),clrcode(0),

def margenta(str): return MARGENTA + str + RESET
def green(str): return GREEN + str + RESET
def blue(str): return BLUE + str + RESET
def yellow(str): return YELLOW + str + RESET

def print_margenta(str): print(MARGENTA + str + RESET)
def print_green(str): print(GREEN + str + RESET)
def print_blue(str): print(BLUE + str + RESET)
def print_yellow(str): print(YELLOW + str + RESET)
