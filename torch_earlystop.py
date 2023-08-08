import sys
sys.path.append('.')
import numpy as np

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metric_value):
        if self.best is None:
            self.best = metric_value
            return False

        if np.isnan(metric_value):
            return True

        if self.is_better(metric_value, self.best):
            self.num_bad_epochs = 0
            self.best = metric_value
            print(f'--OK new best: {self.best}')
        else:
            self.num_bad_epochs += 1
            print(f'--bad epochs: {self.num_bad_epochs}, cur best: {self.best:.4f}')

        if self.num_bad_epochs >= self.patience:
            print('terminating because of early stopping!')
            return True

        return False

    def __str__(self):
        return f'{type(self).__name__}: mode={self.mode}, patience={self.patience}'

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda newval, best: newval < (best - min_delta)
            if mode == 'max':
                self.is_better = lambda newval, best: newval > (best + min_delta)
        else:
            if mode == 'min':
                self.is_better = lambda newval, best: newval < best - (best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda newval, best: newval > best + (best * min_delta / 100)



if __name__ == '__main__':
    ea = EarlyStopping()
    print(ea)