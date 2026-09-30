from .arrays import *
from .bc_evaluator import BCEvaluator
from .bc_training import *
from .colab import *
from .config import *
from .data_encoder import *
from .evaluator import MADEvaluator
from .mamujoco_rendering import MAMuJoCoRenderer
from .mpe_rendering import MPERenderer, NullRenderer
from .offline_evaluator import MADOfflineEvaluator
from .progress import *
from .rendering import *
from .serialization import *
from .setup import *
from .smac_rendering import SMACRenderer
from .training import *


def __getattr__(name):
    # MPE and SMAC must not import the optional MuJoCo renderer.
    if name == "MAHalfCheetahRenderer":
        from .mahalfcheetah_rendering import MAHalfCheetahRenderer
        globals()[name] = MAHalfCheetahRenderer
        return MAHalfCheetahRenderer
    raise AttributeError(name)
