from pathlib import Path
from torch import cuda

class GlobalConfig():
    def __init__(self):
        self.__dataset_path = r"H:\ArmorId"
        self.__device = "cuda:0" if cuda.is_available() else "cpu"
        self.__model_name = "MobileNetv3_small"
        self.__classes_num = 5
        self.__input_size = (20, 28)
        self.__wid_mul=1
        self.__learn_rate=0.02
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
    def INPUT_SIZE(self):
        return self.__input_size
    def WID_MUL(self):
        return self.__wid_mul
    def LR(self):
        return self.__learn_rate

global_config = GlobalConfig()