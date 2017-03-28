
import heapq


class pqueue(object):
    def __init__(self, maxsize, max=False):
        self.queue = []
        self.maxsize = maxsize
        self.max = max

    def push(self, item, priority):
        if self.max:
            priority = -priority
        heapq.heappush(self.queue, (priority, item))
        if len(self.queue) > self.maxsize:
            self.pop()

    def pop(self):
        return heapq.heappop(self.queue)

    def __len__(self):
        return len(self.queue)

    def get_min(self):
        if self.max:
            p, x = max(self.queue)
            return -p, x
        return min(self.queue)

    def get_max(self):
        if self.max:
            p, x = min(self.queue)
            return -p, x
        return max(self.queue)

    def is_full(self):
        return len(self) == self.maxsize


class EarlyStoppingException(Exception):
    def __init(self, message, data={}):
        super(EarlyStopping, self).__init__(message)
        self.message = message
        self.data = data


class EarlyStopping(pqueue):
    def __init__(self, maxsize):
        super(EarlyStopping, self).__init__(maxsize, max=True)

    def add_checkpoint(self, checkpoint, model=None):
        self.push(model, checkpoint)
        if self.is_full():
            smallest, model = self.get_min()
            if checkpoint > smallest:
                message = ("Stopping after %d checkpoints. " % self.maxsize)
                message += "Best score [%g]" % smallest
                raise EarlyStoppingException(
                    message, {'model': model, 'smallest': smallest})
            else:
                self.queue = []
