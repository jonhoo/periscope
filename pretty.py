from termcolor import colored
import datetime
import sys

def section(msg):
    print(colored("\n::", "blue", attrs=["bold"]), colored(msg, attrs=["bold"]), file=sys.stderr)
def task(msg):
    print(colored("==>", "green", attrs=["bold"]), colored(msg, attrs=["bold"]), file=sys.stderr)
def subtask(msg):
    print(colored(" ->", "blue", attrs=["bold"]), colored(msg, attrs=["bold"]), file=sys.stderr)

from progressbar import Bar, SimpleProgress, Percentage, ProgressBar, Timer

class AbsoluteETABrief(Timer):
    '''Variation of progressbar.AbsoluteETA which is smaller for 80cols.'''

    def _eta(self, progress, data, value, elapsed):
        """Update the widget to show the ETA or total time when finished."""
        if value == progress.min_value:  # pragma: no cover
            return 'ETA: --:--:--'
        elif progress.end_time:
            return 'Fin: %s' % self._format(progress.end_time)
        else:
            eta = elapsed * progress.max_value / value - elapsed
            now = datetime.datetime.now()
            eta_abs = now + datetime.timedelta(seconds=eta)
            return 'ETA: %s' % self._format(eta_abs)

    def _format(self, t):
        return t.strftime("%H:%M:%S")

    def __call__(self, progress, data):
        '''Updates the widget to show the ETA or total time when finished.'''
        return self._eta(progress, data, data['value'],
                         data['total_seconds_elapsed'])


def progress(number, **kwargs):
    return ProgressBar(max_value=number, widgets=[Percentage(), ' (', SimpleProgress(), ') ', Bar(), ' ', Timer(), ' ', AbsoluteETABrief()], **kwargs).start()
