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
# ==============================================================================
"""
dataset
"""
import numpy as np

from mindelec.geometry import Cuboid, create_config_from_edict
from mindelec.data import Dataset,ExistedDataConfig

from src.config import sampling_config

import pandas as pd
 
 
# prepare train input

# define geometry

def pre_csv_to_npy(Maxwell_3d_config):

    csv_data_path = Maxwell_3d_config['csv_data_path']
    pdata = pd.read_csv(csv_data_path)
    #前两列代表的是坐标 后6列对应的不同eigenmode下的波导面
    #参考modulus例子 选取 u0 waveguide_port_outvar_numpy = {"uz": data_var["u0"]}
    npdata = pd.DataFrame(pdata).values
    # print(npdata)
  
    
    waveguide_y = npdata[:,0:1]
    waveguide_z = npdata[:,1:2]
    waveguide_x = Maxwell_3d_config['coord_min'][0]*np.ones_like(waveguide_y)
    waveguide_data = np.concatenate([waveguide_x,waveguide_y,waveguide_z],axis=-1)

    np.save("waveguide_data.npy", waveguide_data) 
    np.save("waveguide_label.npy", npdata[:,2:3]) 

    


def create_train_dataset(Maxwell_3d_config):
    

    cuoid_space = Cuboid(name=Maxwell_3d_config["geom_name"],
                        coord_min=Maxwell_3d_config["coord_min"],
                        coord_max=Maxwell_3d_config["coord_max"],
                        sampling_config=create_config_from_edict(sampling_config))

    geom_dict = {cuoid_space: ["domain","BC"]} 

    pre_csv_to_npy(Maxwell_3d_config)

    data_config = ExistedDataConfig(name='waveguide',
                                    data_dir=['waveguide_data.npy','waveguide_label.npy'],
                                    columns_list=['data','label'], data_format="npy", constraint_type="Function")



    train_dataset = Dataset(geometry_dict=geom_dict,existed_data_list=[data_config])

    return train_dataset




# prepare test input and label
def test_data_prepare(config):
    """create test dataset"""
    coord_min = config["coord_min"]
    coord_max = config["coord_max"]
    axis_size = config["axis_size"]
    wave_number = config.get("wave_number", 2.0)

    # input
    axis_x = np.linspace(coord_min[0], coord_max[0], num=axis_size, endpoint=True)
    axis_y = np.linspace(coord_min[1], coord_max[1], num=axis_size, endpoint=True)
    axis_z = np.linspace(coord_min[2], coord_max[2], num=axis_size, endpoint=True)
    mesh_x, mesh_y, mesh_z = np.meshgrid(axis_y, axis_x, axis_z,indexing='ij')
    input_data = np.hstack((mesh_x.flatten()[:, None], mesh_y.flatten()[:, None], mesh_z.flatten()[:, None])).astype(np.float32)

    # # label
    # label = np.zeros((axis_size, axis_size, 1))
    # for i in range(axis_size):
    #     for j in range(axis_size):
    #         label[i, j, 0] = np.sin(wave_number * axis_x[j])

    # label = label.reshape(-1, 1).astype(np.float32)

    return input_data #, label
