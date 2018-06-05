
import logging

__version__ = "0.3"

# grab git commit of seqmod
from .misc.git import GitInfo

try:
    __commit__ = GitInfo(__file__).get_commit()
except Exception:
    logging.warning("`seqmod` is not git-tracked, I won't report seqmod git commit.")
    __commit__ = 'Unknown!'

from .misc import *
from .hyper import *
from .modules import *
from . import utils
from . import loaders

# shortcut (still keep the old import for backward compatibility)
from seqmod.misc.beam_search import Beam
from seqmod.misc.dataset import Dict, MultiDict, PairedDataset
from seqmod.misc.early_stopping import EarlyStopping, EarlyStoppingException
from seqmod.misc.loggers import StdLogger, VisdomLogger
from seqmod.misc.preprocess import text_processor
from seqmod.misc.trainer import Trainer
