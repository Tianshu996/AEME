from math import inf


def _init_is_better(mode, threshold_mode):
    if mode not in {'min', 'max'}:
        raise ValueError('mode ' + mode + ' is unknown!')
    if threshold_mode not in {'rel', 'abs'}:
        raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')
    if mode == 'min':
        mode_worse = inf
    else: 
        mode_worse = -inf
    return mode_worse


def is_better(a, best, mode, threshold_mode, threshold):
    if mode == 'min' and threshold_mode == 'rel':
        rel_epsilon = 1. - threshold
        return a < best * rel_epsilon
    elif mode == 'min' and threshold_mode == 'abs':
        return a < best - threshold
    elif mode == 'max' and threshold_mode == 'rel':
        rel_epsilon = threshold + 1.
        return a > best * rel_epsilon
    else:  
        return a > best + threshold


class AdaIter:
    def __init__(self, mode='min', infactor=1, patience=5,
                 threshold=1e-3, threshold_mode='rel',
                 iter_term=1, max_iter=10, verbose=True,
                 early_stop_threshold=1e-4):
        """
        Initialize the adaptive iteration controller with early stopping.

        Args:
            mode (str): 'min' or 'max' - whether to consider lower or higher values as better
            infactor (float): Amount to increase iteration term by
            patience (int): Number of epochs to wait before increasing iterations
            threshold (float): Minimum change threshold to consider as improvement
            threshold_mode (str): 'rel' or 'abs' for relative or absolute threshold
            iter_term (int): Initial iteration term
            max_iter (int): Maximum allowed iteration term
            verbose (bool): Whether to print iteration changes
            early_stop_threshold (float): Threshold for early stopping
        """
        if infactor <= 0:
            raise ValueError('Factor should be > 0.')

        self.factor = infactor
        self.patience = patience
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.iter_term = iter_term
        self.max_iter = max_iter
        self.verbose = verbose
        self.early_stop_threshold = early_stop_threshold
        self.mode_worse = _init_is_better(self.mode, self.threshold_mode)
        self.last_epoch = 0
        self.best = self.mode_worse
        self._reset()
        self.should_stop = False

    def _reset(self):
        self.best = self.mode_worse
        self.num_bad_epochs = 0

    def check_early_stop(self, metrics):

        if self.mode == 'min' and metrics <= self.early_stop_threshold:
            if self.verbose:
                print(f'Early stopping: validation loss {metrics} reached threshold {self.early_stop_threshold}')
            return True
        elif self.mode == 'max' and metrics >= self.early_stop_threshold:
            if self.verbose:
                print(f'Early stopping: validation metric {metrics} reached threshold {self.early_stop_threshold}')
            return True
        elif self.iter_term >= self.max_iter:
            if self.verbose:
                print(f'Early stopping: reached maximum iterations {self.max_iter}')
            return True
        return False

    def step(self, metrics, epoch=None):

        current = float(metrics)

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        self.should_stop = self.check_early_stop(current)
        if self.should_stop:
            return self.iter_term

        if is_better(current, self.best, self.mode, self.threshold_mode, self.threshold):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self._increase_iter(epoch)
            self.num_bad_epochs = 0

        return self.iter_term

    def _increase_iter(self, epoch):

        old_iter = self.iter_term
        self.iter_term = min(self.iter_term + self.factor, self.max_iter)

        if self.verbose:
            print(f'Epoch {epoch:5d}: increasing iterations from {old_iter:5.1f} to {self.iter_term:5.1f}')