### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.Color_Input = opt.Color_Input
        self.Color_Output = opt.Color_Output

        ### input A (label maps)
        dir_A = '_A'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)

        dir_B = '_B'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)

        transform_A = get_transform(self.opt, params,Color_Input=self.Color_Input)
        if self.Color_Input=="RGB":
            A_tensor= transform_A(A.convert('RGB'))
        elif self.Color_Input == "gray":
            A_tensor = transform_A(A.convert('I'))
        else:
            raise NotImplementedError("This Color Channel not implemented")
        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)

        B_path = self.B_paths[index]
        transform_B = get_transform(self.opt, params,Color_Input=self.Color_Output)
        B = Image.open(B_path)
        if self.Color_Output == "RGB":
            B_tensor = transform_B(B.convert("RGB"))
        elif self.Color_Output == "gray":
            B_tensor = transform_B(B.convert("I"))
        else:
            raise NotImplementedError("This Color Channel not implemented")

        input_dict = {'label': A_tensor,'image': B_tensor, 'path': A_path, 'index':index}

        return input_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'