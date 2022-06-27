import time


class Timer(object):
    """A simple timer."""

    def __init__(self):
        '''
        Initializes a Timer object.
        '''

        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        '''
        Starts the timer.
        '''

        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        '''
        Stops the timer.

        Keyword Arguments:
            average {boolean} -- determines whether the return value is the
                average time across all calls or only the last call

        Returns:
            float -- the average time across all calls or only the last call
        '''

        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff
