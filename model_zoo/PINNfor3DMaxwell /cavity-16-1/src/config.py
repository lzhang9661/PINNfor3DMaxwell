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
        # 'random_sampling': False,
        # 'size': [64,64,64],
        # 'sampler':'uniform',
        'random_sampling': True,
        'size': 1024*300,
    }),
    'BC': ed({
        'random_sampling': True,
        'size': 1024*300,
        'sampler':'uniform',
    })
})

# config
Maxwell_3d_config = {
    "name": "Maxwell3D",
    "geom_name": "cuboid",
    "input_size": 3,
    "output_size": 3, 
    "Epochs": 3000,
    "batch_size": 1024,
    "lr": 0.001,
    "coord_min": [0.0, 0.0, 0.0],
    "coord_max": [2.0, 2.0, 2.0],
    "axis_size": 51,
    "wave_number": 16,
    "eigenmode": 1,
    "weight_PEC":100,
    "weight_ABC":10,
    "weight_waveguide":100,
    "weight_bc":100,

    "layers" : 6,
    "neurons" : 32,

}
