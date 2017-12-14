
import heapq


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
        super(EarlyStopping, self).__init__(message)
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
    maxsize: int, buffer size
        Only consider so many previous checkpoints before raising the
        Exception, buffer will be freed after `maxsize` checkpoints are
        introduced. After freeing the buffer the previously best checkpoint is
        kept in the buffer to allow for comparisons with checkpoints that are
        far in the past. The number of failed attempts will however be freed
        alongside the buffer.

    patience: int (optional, default to maxsize)
        Number of failed attempts to wait until finishing training.

    reset_patience: bool, default True

    """

    def __init__(self, maxsize, tolerance=1e-4,
                 patience=None, reset_patience=True):
        """Set params."""
        self.tolerance = tolerance
        self.patience, self.fails = patience or maxsize - 1, 0
        self.reset_patience = reset_patience
        self.stopped = False
        self.checks = []  # register losses over checkpoints

        if self.patience >= maxsize:
            raise ValueError("patience must be smaller than maxsize")

        super(EarlyStopping, self).__init__(maxsize, heapmax=True)

    def find_smallest(self):
        return sorted(enumerate(self.checks), key=lambda i: i[1])[-1][0]

    def add_checkpoint(self, checkpoint, model=None):
        """Add loss to queue and stop if patience is exceeded."""
        self.push(model, checkpoint)
        self.checks.append(checkpoint)
        smallest, model = self.get_min()

        if checkpoint + self.tolerance > smallest:
            self.fails += 1
            if self.fails == self.patience:
                self.stopped = True
                msg = "Stop after {} checkpoints. ".format(self.patience)
                msg += "Best score {:.3f} ".format(smallest)
                msg += "at checkpoint {}.".format(self.find_smallest())
                raise EarlyStoppingException(
                    msg, {'model': model,
                          'smallest': smallest})

        if self.is_full():
            checkpoint, model = self.get_min()
            self.queue = []
            if self.reset_patience:
                self.fails = 0
            self.add_checkpoint(checkpoint, model)
