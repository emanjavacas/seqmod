
import os
import logging
from subprocess import check_output, CalledProcessError
from contextlib import contextmanager
import sys


@contextmanager
def silence(stderr=True, stdout=True):
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        sys.stderr = devnull
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr
            sys.stdout = old_stdout


class GitInfo:
    """
    Utility class to retrieve git-based info from a repository
    """
    def __init__(self, fname):
        if os.path.isfile(fname):
            self.dirname = os.path.dirname(fname)
        elif os.path.isdir(fname):
            self.dirname = fname
        else:
            # not a file
            self.dirname = None

        if not os.path.isfile(fname) and not os.path.isdir(fname):
            logging.warn("[GitInfo]: Input file doesn't exit")

        else:
            try:
                with silence():
                    check_output(['git', '--version'], cwd=self.dirname)
            except FileNotFoundError:
                self.dirname = None
                logging.warn("[GitInfo]: Git doesn't seem to be installed")
            except CalledProcessError as e:
                self.dirname = None
                code, _ = e.args
                if code == 128:
                    logging.warn("[GitInfo]: Script not git-tracked")
                else:
                    logging.warn("[GitInfo]: Unrecognized git error")

    def run(self, cmd):
        if self.dirname is None:
            return

        return check_output(cmd, cwd=self.dirname).strip().decode('utf-8')

    def get_commit(self):
        """
        Returns current commit on file or None if an error is thrown by git
        (OSError) or if file is not under git VCS (CalledProcessError)
        """
        return self.run(["git", "describe", "--always"])

    def get_branch(self):
        """
        Returns current active branch on file or None if an error is thrown
        by git (OSError) or if file is not under git VCS (CalledProcessError)
        """
        return self.run(["git", "rev-parse", "--abbrev-ref", "HEAD"])

    def get_tag(self):
        """
        Returns current active tag
        """
        return self.run(["git", "describe", "--tags", "--abbrev=0"])
