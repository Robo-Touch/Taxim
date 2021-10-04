import numpy as np
class RawData:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        data_file = np.load(dataPath, allow_pickle=True)

        self.f0 = data_file['f0']
        self.imgs = data_file['imgs']
        self.touch_center = data_file['touch_center']
        self.touch_radius = data_file['touch_radius']
        self.names = data_file['names']
        self.img_size = data_file['img_size']
