"""
Default Exp form the YOLOX inference.
"""
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    """
    Experiment class for YOLOX model.

    Copied from "https://github.com/DanceTrack/DanceTrack/blob/main/ByteTrack/exps/example/dancetrack/yolox_x.py"
    """
    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.class_names = ('pedestrian',)
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split('.')[0]
        self.train_ann = 'train.json'
        self.val_ann = 'val.json'
        self.test_ann = 'test.json'

        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.random_size = (18, 32)
        self.max_epoch = 8
        self.print_interval = 20
        self.eval_interval = 5
        self.test_conf = 0.1
        self.nmsthre = 0.7
        self.no_aug_epochs = 1
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1


DEFAULT_EXP_PATH = __file__
DEFAULT_EXP_NAME = Exp.__name__
