
import heapq
from operator import itemgetter


class pqueue(object):
    def __init__(self, maxsize, heapmax=False):
        self.queue = []
        self.maxsize = maxsize
        self.heapmax = heapmax

    def push(self, item, priority):
        if self.heapmax:
            priority = -priority
        heapq.heappush(self.queue, (priority, item))
        if len(self.queue) > self.maxsize:
            # print("Popped {}".format(self.pop()))
            self.pop()

    def pop(self):
        p, x = heapq.heappop(self.queue)
        if self.heapmax:
            return -p, x
        return p, x

    def __len__(self):
        return len(self.queue)

    def get_min(self):
        if self.heapmax:
            p, x = max(self.queue)
            return -p, x
        else:
            p, x = min(self.queue)
            return p, x

    def get_max(self):
        if self.heapmax:
            p, x = min(self.queue)
            return -p, x
        else:
            p, x = max(self.queue)
            return p, x

    def is_full(self):
        return len(self) == self.maxsize

    def is_empty(self):
        return len(self.queue) == 0


class EarlyStoppingException(Exception):
    def __init(self, message, data={}):
        super(EarlyStoppingException, self).__init__(message)
        self.message = message
        self.data = data


class EarlyStopping(pqueue):
    """Queue-based EarlyStopping that caches previous versions of the models.

    Early stopping takes place if perplexity increases a number of times
    higher than `patience` over the lowest recorded one without resulting in
    the buffer being freed. On buffer freeing, the number of fails is reset but
    the lowest recorded value is kept. The last behaviour can be tuned by
    passing reset_patience equal to False.

    Parameters
    ----------
    patience: int (optional, default to maxsize)
        Number of failed attempts to wait until finishing training.

    maxsize: int, buffer size,
        Number of checks after which the queue gets emptied. (After emptying
        the best model and its corresponding checkpoint are still kept).
        Useful to control the amount of models that are stored in memory.

    tolerance: float, minimum improvement needed for a checkpoint to not be
        considered a failure.

    reset_patience: bool, default True,
        Whether to reset the number of registered failures when encountering
        a new lowest checkpoint.

    reset_on_emptied: bool, default False,
        Whether to reset the number of registered failures after emptying the
        queue.
    """

    def __init__(self, patience, maxsize=10, tolerance=1e-4,
                 reset_patience=True, reset_on_emptied=False):
        """Set params."""
        self.patience = patience
        self.maxsize = maxsize
        self.tolerance = tolerance
        self.reset_patience = reset_patience
        self.reset_on_emptied = reset_on_emptied
        # data
        self.stopped = False
        self.fails = 0
        self.checks = []  # register losses over checkpoints

        if self.reset_on_emptied and patience >= maxsize:
            raise ValueError(
                "`patience` must be smaller than maxsize when resetting"
                " patience on full queue (`reset_on_emptied`)")

        super(EarlyStopping, self).__init__(self.maxsize, heapmax=True)

    def _find_smallest(self):
        (index, _), *_ = sorted(enumerate(self.checks), key=itemgetter(1))
        return index + 1        # 1-index

    def _build_message(self, smallest):
        msg = "Stop after {} checkpoints. ".format(len(self.checks))
        msg += "Best score {:.4f} ".format(smallest)
        msg += "at checkpoint {} ".format(self._find_smallest())
        msg += "with patience {}.".format(self.patience)
        return msg

    def add_checkpoint(self, checkpoint, model=None, add_check=True):
        """Add loss to queue and stop if patience is exceeded."""
        if add_check:
            self.checks.append(checkpoint)

        if self.is_empty():
            self.push(model, checkpoint)
            return

        smallest, best_model = self.get_min()
        if self.is_full():
            self.queue = []
            if self.reset_on_emptied:
                self.fails = 0
            if self.patience > 1:  # avoid loop
                self.add_checkpoint(
                    checkpoint, model=best_model, add_check=False)

        if (checkpoint + self.tolerance) >= smallest:
            self.fails += 1
        else:
            if self.reset_patience:
                self.fails = 0

        if self.fails == self.patience:
            self.stopped = True
            raise EarlyStoppingException(
                self._build_message(smallest),
                {'model': best_model, 'smallest': smallest})

        self.push(model, checkpoint)
