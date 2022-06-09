import signal 
from contextlib import contextmanager

# A class that allows us to check the execution time of a method, and if takes 
# too long, throws an error - useful for predict(), where sometimes there were 
# numbers so large that python crashed computationally
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
