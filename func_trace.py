
from pathlib import Path
import glob
import sys
import io, os
import inspect
from dataclasses import dataclass
import re
from types import FrameType
from typing import Dict, Union, Optional, Set
import builtins
import threading
import enum

GRAY, RED, GREEN, YELLOW, RESET = '\33[30m', '\33[31m', '\33[32m', '\33[33m', '\33[0m' # logging colors
# GRAY, RED, GREEN, YELLOW, RESET = '', '', '', '', '' # no clr

MAX_FUNC_LOGGING = 20 # 999999


@enum.unique
class _Suppress(enum.Enum):
    SelfOnly = enum.auto()
    Following = enum.auto()
    SelfAndFollowing = enum.auto()


class Trace:
    _workdir = None
    _builtin_print = builtins.print
    Suppress = _Suppress

    @dataclass
    class FuncContext:
        # id:str
        show_args:int = 1
        logging_cnt:int=MAX_FUNC_LOGGING
        suppress:Optional[_Suppress] = None

    _FuncCxt:Dict[str, FuncContext] = dict()

    @staticmethod
    def start(indent_print:bool = False):
        print('start Trace...')
        assert len(Trace._whitelist_files) > 0, "No whitelist files"
        Trace.rmv_sources(__file__) # remove self.

        # common setup: non ROI funcs
        Trace.set_func_context('<module>', suppress=Trace.Suppress.SelfAndFollowing) # module loading
        for fname in ['<listcomp>', '<genexpr>', '<dictcomp>']:
            Trace.set_func_context(fname, suppress=Trace.Suppress.SelfOnly)

        @dataclass
        class TlsData:
            name:str= threading.current_thread().name
            suppress_from:int = 0
            suppress:Trace.Suppress=None
            indent:str=''

        this_thread = threading.current_thread()
        assert getattr(this_thread, 'tlsdata', None) is None
        setattr(this_thread, 'tlsdata', TlsData())

        sys.setprofile(Trace._trace)
        if indent_print:
            builtins.print = Trace.log


    @staticmethod
    def set_func_context(func_name:str, srcpath:str=None, *,
                         show_args:int=1,
                         logging_cnt:int=MAX_FUNC_LOGGING,
                         suppress:_Suppress=None):
        assert Path(Trace._workdir).is_dir()

        id = func_name
        if srcpath is not None:
            # func only in srcpath.
            id += "@" + os.path.abspath(srcpath)

        Trace._FuncCxt[id] = Trace.FuncContext(show_args=show_args, logging_cnt=logging_cnt, suppress=suppress)
        print(f'+{id}: {show_args=}, {logging_cnt=}, {suppress=}')

    @staticmethod
    def set_func_context2(funcstr:str, *,
                          show_args:int=1,
                          logging_cnt:int=MAX_FUNC_LOGGING,
                          suppress:_Suppress=None):
        '''
        funcstr: you can removed the line with python comment.
            # ex) get func_name and srcpath from ' IBasicBlock.forward() models/arcface.py:61, from '
        '''
        funcstr = re.sub(r'^\s*#.*$', '', funcstr, flags=re.MULTILINE) # remove comment(#) line
        matches = re.findall(r'([\w<](?!\(\)).+)\(\)\s+([^:]+):\d+', funcstr)
        if matches:
            for match in matches:
                func_name, srcpath = match
                func_name = func_name.strip()
                srcpath = srcpath.strip()
                Trace.set_func_context(func_name, srcpath,
                                         show_args=show_args, logging_cnt=logging_cnt, suppress=suppress)

    _whitelist_files:Set[str] = set()      # Look for these words in the file path.

    @staticmethod
    def add_sources(items:Union[str,list,set], all_subdir:bool=False):
        if Trace._workdir is None:
            Trace._workdir = os.path.abspath(os.getcwd())
            if Trace._workdir[-1] != '/': Trace._workdir += '/'
            print('working dir:' + Trace._workdir)

        if not isinstance(items, (list,set)):
            items = [items]

        for i in items:
            if Path(i).is_dir():
                exts = '**/*.py' if all_subdir else '*.py'
                folder = os.path.abspath(i)
                srcs = glob.glob( os.path.join(folder, exts), recursive=all_subdir)
                # srcs = [Trace._trim_workingdir(i) for i in srcs if '__init__.py' not in i] # remove __init__.py
                srcs = [i for i in srcs if '__init__.py' not in i] # remove __init__.py
                Trace._whitelist_files.update(srcs)
                print(f'+ {len(srcs)} for {i}')
                # for i in srcs: print('   -', i)

            elif Path(i).is_file():
                # file = Trace._trim_workingdir( os.path.abspath(i) )
                file = os.path.abspath(i)
                Trace._whitelist_files.add(file)
                # print('+', i)


    @staticmethod
    def rmv_sources(items:Union[str,list,set]):
        if not isinstance(items, (list,set)):
            items = [items]

        for i in items:
            if Path(i).is_dir():
                all_subdir = False
                folder = os.path.abspath(i)
                exts = '**/*.py' if all_subdir else '*.py'
                srcs = glob.glob( os.path.join(folder, exts), recursive=all_subdir)
                # srcs = [Trace._trim_workingdir(i) for i in srcs]
                Trace._whitelist_files.discard(srcs)
                print(f'+ {len(srcs)} for {i}')
                # print(f'+ <{srcs[0]}>') # debug

            elif Path(i).is_file():
                # file = Trace._trim_workingdir( os.path.abspath(i) )
                file = os.path.abspath(i)
                Trace._whitelist_files.discard(file)


    @staticmethod
    def rmv_sources2(filestrs:str):
        filestrs = re.sub(r'^\s*#.*$', '', filestrs, flags=re.MULTILINE) # remove comment(#) line
        py_files = re.findall(r'(\S+\.py)', filestrs)
        Trace.rmv_sources(py_files)

    #
    # function arguments
    #
    @staticmethod
    def _args(frame:FrameType):
        args,_,_,values = inspect.getargvalues(frame)
        for i,name in enumerate(args):
            if name == 'self': continue
            value = values[name]

            if ".npyio." in str(type(value)): value = value.item()

            if type(value) == dict:
                Trace.log(f"- args{i}. {name}=", clr=YELLOW)
                for k,v in value.items():
                    Trace.log(f"  - {k}: {v.shape if hasattr(v, 'shape') else v}", clr=YELLOW)

            elif hasattr(value, 'shape'): # tensor or np.array
                Trace.log(f"- args{i}. {name}= {value.shape}", clr=YELLOW)

            #
            # TODO - add some codes to support another type.
            #
            else:
                Trace.log(f"- args{i}. {name}= {value}", clr=YELLOW)


    @staticmethod
    def args():
        frame = inspect.currentframe().f_back
        if frame.f_code.co_filename not in Trace._whitelist_files:
            return

        tls = threading.current_thread().tlsdata
        if tls.suppress is not None:
            return

        callee_func, callee_file, callee_line = Trace._getfi(frame)

        funcxt = Trace._FuncCxt.get(callee_func, None) # without file scope.
        if not funcxt:
            fnid = f"{callee_func}@{callee_file}"
            funcxt = Trace._FuncCxt.get(fnid, Trace.FuncContext())

        if funcxt.suppress == _Suppress.SelfOnly:
            return
        if funcxt.logging_cnt < 0:
            return
        Trace._args(frame)


    @staticmethod
    def _trim_workingdir(path):
        if Trace._workdir in path:
            return path[len(Trace._workdir):]
        return path


    @staticmethod
    def _getfi(frame):
        if frame is None: return ('?', '?', -1)
        src_path = frame.f_code.co_filename
        # src_path = Trace._trim_workingdir(frame.f_code.co_filename)

        func_name = frame.f_code.co_name
        if 'self' in frame.f_locals:
            for cls in inspect.getmro(frame.f_locals['self'].__class__):
                if hasattr(cls, func_name):
                    member = getattr(cls, func_name)
                    if inspect.isfunction(member) and member.__code__ == frame.f_code:
                        return (member.__qualname__, src_path, frame.f_lineno)

            func_name = frame.f_locals['self'].__class__.__name__ + '.' + func_name
        return (func_name, src_path, frame.f_lineno)


    @staticmethod
    def _trace(frame, event, arg):
        if event != "call" and event != 'return': return

        # filtering non ROI
        if frame.f_code.co_filename not in Trace._whitelist_files:
            return

        callee_func, callee_file, callee_line = Trace._getfi(frame)
        fnid = f"{callee_func}@{callee_file}"

        # if fnid in Trace._blacklist_funcs:
        #     return

        tls = threading.current_thread().tlsdata
        if event == "call":
            if tls.suppress is not None:
                return

            fncxt = Trace._FuncCxt.get(callee_func, None) # without file scope.
            if not fncxt:
                fncxt = Trace._FuncCxt.get(fnid, Trace.FuncContext())
                Trace._FuncCxt[fnid] = fncxt

            if fncxt.suppress is not None:
                if fncxt.suppress == _Suppress.SelfOnly:
                    return
                tls.suppress_from = id(frame)
                tls.suppress = fncxt.suppress
                if fncxt.suppress == _Suppress.SelfAndFollowing:
                    return

            fncxt.logging_cnt -= 1
            if fncxt.logging_cnt < 0:
                return

            caller_func, caller_file, caller_line = Trace._getfi(frame.f_back)
            Trace.log(f'{{ {callee_func}() { Trace._trim_workingdir(callee_file)}:{callee_line}, '
                        f'from {caller_func} ({Trace._trim_workingdir(caller_file)}:{caller_line}) - {tls.name}',
                        clr=GREEN)
            tls.indent += '\t'
            # print('debug: enter', id(frame))

            if fncxt.show_args>0:
                fncxt.show_args -= 1
                Trace._args(frame)

        elif event == "return":

            if tls.suppress:
                if tls.suppress_from != id(frame):
                    return
                suppress = tls.suppress
                tls.suppress = None
                if suppress == _Suppress.SelfAndFollowing:
                    return

            fncxt = Trace._FuncCxt.get(callee_func, None) # without file scope.
            if not fncxt:
                fncxt = Trace._FuncCxt.get(fnid, Trace.FuncContext())
            if fncxt.suppress == _Suppress.SelfOnly:
                return

            if fncxt.logging_cnt >= 0:
                # print('debug: exit', id(frame))
                tls.indent = tls.indent[:-1]
                caller_frame = Trace.get_user_frame(frame.f_back)
                caller_func, caller_file, caller_line = Trace._getfi(caller_frame)
                Trace.log(f'}} {callee_func}(), '
                          f'return to {caller_func} ({Trace._trim_workingdir(caller_file)}:{caller_line}) - {tls.name}', clr=GREEN)

    @staticmethod
    def get_user_frame(frame):
        # return frame # disable return to user frame
        iter = frame
        while iter is not None:
            if Trace._workdir in iter.f_code.co_filename:
                return iter
            iter = iter.f_back
        return frame

    #
    # logging funcs
    #
    # @staticmethod
    # def fmt(*args, **kwargs):
    #     buf = io.StringIO()
    #     print(*args, **kwargs, end='', file=buf)
    #     message = buf.getvalue().replace('\n', '\n' + Trace.indent)
    #     buf.close()
    #     return Trace.indent + message


    logfp:io.FileIO = None
    @staticmethod
    def set_logfile(logpath:Union[str,Path]):
        Trace.logfp = open(str(logpath), 'w', buffering=1) # line buffer

    @staticmethod
    def log(*args, clr:Optional[str]=None, **kwargs):

        tls = threading.current_thread().tlsdata

        end = kwargs.pop('end', '\n')
        buf = io.StringIO()
        Trace._builtin_print(*args, **kwargs, end='', file=buf)
        message = buf.getvalue()
        if message[-1] == '\n':
            message = tls.indent + message[:-1].replace('\n', '\n' + tls.indent) + '\n' # do not indent last '\n'
        else:
            message = tls.indent + message.replace('\n', '\n' + tls.indent)
        buf.close()

        if clr is None:
            Trace._builtin_print(message, end=end)
        else:
            Trace._builtin_print(clr, message, RESET, end=end)

        if Trace.logfp is not None:
            Trace._builtin_print(message, file = Trace.logfp, end=end)




# #
# # examples
# #
# class Base:
#     def __init__(self, b) -> None:
#         print('b', b, ' in Base')
#         pass

# class A(Base):

#     def __init__(self, a=1, b=0) -> None:
#         super().__init__(b)
#         pass


#     def memberfunc(self, a,b):
#         print('memberfunc', a,b)


# def scope_test():

#     print('-------------- max_count=2 test ---------------')
#     def max_count_test(a,b,c):
#         print(f'It will shown only max_count(2) times', a,b,c)

#         obj = A(100,101)
#         obj.memberfunc(4,5)
#         return a+b+c

#     for _ in range(5):
#         max_count_test(1,2,3)

#     print('-------------- suppress test ---------------')

#     def in_suppress_1():
#         print('Not seen: in_suppress_1')

#     def suppress_following():
#         print('enter: suppress_following()')
#         in_suppress_1()
#         print('exit: suppress_following: after in_suppress_1')
#         return

#     suppress_following()
#     return




# if __name__ == '__main__':

#     Trace.add_sources('./')
#     Trace.set_func_context('suppress_following', __file__, suppress=Trace.Suppress.SelfAndFollowing)
#     Trace.set_func_context('suppress_following', __file__, suppress=Trace.Suppress.Following)

#     Trace.start(indent_print=True)
#     Trace.add_sources(__file__)
#     scope_test()

