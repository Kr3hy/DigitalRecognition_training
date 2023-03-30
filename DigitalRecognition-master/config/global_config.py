from pathlib import Path
from torch import cuda

class GlobalConfig():
    def __init__(self):
        self.__dataset_path = r"H:\ArmorId"
        self.__device = "cuda:0" if cuda.is_available() else "cpu"
        self.__model_name = "GDUT_net_withMAP_0"
        self.__classes_num = 8
        self.__input_size = (20,28)
        self.__wid_mul=1
        self.__learn_rate=0.00001
        self.__classes_name=CLASSES_NAME = { 1: 'hero', 2: 'engineer', 3: 'standard3', 4: 'standard4', 5: 'standard5', 6: 'sentry', 7: 'outpost', 8: 'base'}
    @property
    def DATASET_PATH(self):
        assert Path(self.__dataset_path).exists(), NotADirectoryError(f"DATASET_PATH {self.__dataset_path} no found!")
        return self.__dataset_path
    @property
    def DEVICE(self):
        return self.__device
    @property
    def MODEL_NAME(self):
        return self.__model_name
    @property
    def CLASSES_NUM(self):
        return self.__classes_num
    @property
    def CLASSES_NAME(self):
        return self.__classes_name
    @property
    def INPUT_SIZE(self):
        return self.__input_size
    def WID_MUL(self):
        return self.__wid_mul
    def LR(self):
        return self.__learn_rate

global_config = GlobalConfig()