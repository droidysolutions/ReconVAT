from .constants import *
#below line makes MAPS, MAESTRO, MusicNet, Corelli, Application_Wind, Application_Dataset, CustomMusicNet available across your project.
from .dataset import MAPS, MAESTRO, MusicNet, Corelli, Application_Wind, Application_Dataset,CustomDataset
from .decoding import *
from .midi import save_midi
from .utils import *
from .evaluate_functions import *
from .helper_functions import *
# from .Conv_Seq2Seq import *
from .self_attenttion_model import *
from .VAT import *
from .onset_frame_VAT import *
from .UNet_onset import *
from .self_attention_VAT import *
from .Segmentation import *
from .Thickstun_model import *
from .Unet_prestack import *



"""
What is __init__.py?
It makes a directory behave like a package so its files can be imported elsewhere.
It allows importing modules like from model.dataset import CustomMusicNet.
Why Do You Need It?
Without __init__.py, Python won't recognize the model/ directory as a module.
"""