# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed

# config
sampling_config = ed({
    'domain': ed({
        'random_sampling': False,
        'size': [64,64,64],      
        # 'random_sampling': True,
        # 'size': 1024*256,
        # 'sampler':'uniform',
    }),
    'BC': ed({
        'random_sampling': True,
        'size': 64*64*64,
        'sampler':'uniform',
    })
})

# config
Maxwell_3d_config = {
    "name": "Maxwell3D",
    "geom_name": "cuboid",
    "permittivity_in_slab":1.5,
    "permittivity_media":1,
    "len_slab":0.2,
    "input_size": 3,
    "output_size": 3, 
    "Epochs": 3000,
    "batch_size": 512,
    "lr": 0.001,
    "coord_min": [-0.5, -0.5, -0.5],
    "coord_max": [0.5, 0.5, 0.5],
    "axis_size": 51,
    "wave_number": 16,
    # "csv_data_path":"validation/2Dwaveguide_16_1.csv", #替代原来的eigenmode 
    "csv_data_path":"validation/2Dwaveguideport.csv",

    "weight_PEC":100,
    "weight_ABC":10,
    "weight_waveguide":0.5,
    "weight_bc":100,

    

    "layers" : 6,
    "neurons" : 64,

    'save_checkpoint_steps': 1875,
    'keep_checkpoint_max': 150,

}
