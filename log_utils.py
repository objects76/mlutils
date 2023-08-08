
import os
import logging
import sys
import datetime

def curdatetime(type = 'timeonly'):
    type = "" if type is None else type.lower()
    if type == 'timeonly': # timeonly, ex) 04:59:31
        return datetime.datetime.now().strftime("%H:%M:%S")

    if type == 'dayonly': # day only, ex) 20220530
        return datetime.datetime.now().strftime("%Y%m%d")


    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # YYYYMMDD_HHmmSS

class ConsoleFormatter(logging.Formatter):
    # COLORS = {
    #     logging.DEBUG   : "\033[38;20m", # grey
    #     logging.INFO    : "\033[92;20m", # green
    #     logging.WARNING : "\033[33;20m", # yellow
    #     logging.ERROR   : "\033[31;20m", # red
    #     logging.CRITICAL: "\033[31;1m", # bold_red
    # }
    # def __init__(self):
    #     FORMATS = "%(asctime)s - %(message)s at %(funcName)s() %(filename)s:%(lineno)d"
    #     super().__init__(FORMATS, datefmt='%Y%m%d-%H:%M:%S')

    # def format(self, record):
    #     msg = super().format(record)
    #     return ConsoleFormatter.COLORS[record.levelno] + msg + "\033[0m"
    green = "\033[{}m".format(92)
    grey = "\033[{}m".format(38)
    yellow = "\033[{}m".format(33)
    red = "\033[{}m".format(31)
    bold_red = "\033[{};1m".format(31)
    reset = "\033[{}m".format(0)
    format = "%(asctime)s - %(message)s"
    format_d = "%(asctime)s - %(message)s at %(funcName)s() %(filename)s:%(lineno)d"

    FORMATS = {
        logging.DEBUG: green + format_d + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


_inialized = []
def get_logger(name='default'):
    if name in _inialized:
        return logging.getLogger(name)
    return None

def init_log(name='default', logfile_path=None, enable_console=True):
    logger = get_logger(name)
    if logger is not None: return logger

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if enable_console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(ConsoleFormatter())
        console.setLevel(logging.INFO)
        logger.addHandler(console)

    if logfile_path:
        try:
            os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
        except: pass
        formatter = logging.Formatter("[%(process)d.%(threadName)s] %(asctime)s -%(levelname)s: %(message)s", datefmt='%Y%m%d-%H:%M:%S')
        logfile = logging.FileHandler(logfile_path, 'a+','utf-8')
        logfile.setFormatter(formatter)
        logfile.setLevel(logging.INFO)

        logger.addHandler(logfile)

        logger.info('---------------------------------------------------------------------')
        logger.info('----------------------- [ START LOG] --------------------------------')

    global _inialized
    _inialized.append(name)
    return logger

if __name__ == '__main__':
    log = init_log('aa', './aa.log')
    log.info('test ./test.log')
    pass
