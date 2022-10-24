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
from mindelec.data import Dataset
from mindelec.data import ExistedDataConfig

from src.config import sampling_config


# prepare train input

# define geometry

def create_train_dataset(Maxwell_3d_config):
    

    cuoid_space = Cuboid(name=Maxwell_3d_config["geom_name"],
                        coord_min=Maxwell_3d_config["coord_min"],
                        coord_max=Maxwell_3d_config["coord_max"],
                        sampling_config=create_config_from_edict(sampling_config))

    geom_dict = {cuoid_space: ["domain","BC"]} 


    # coord_min = np.array(Maxwell_3d_config["coord_min"])
    # coord_max = np.array(Maxwell_3d_config["coord_max"])

    # BC_names = ["Left","Right","Front","Back","Bottom","Top"]
    # def boundary_data_generate_func(name):

    #     size = sampling_config['bc']['size']
    #     index = [0,1,2]
    #     temp_boundary_points = np.zeros([size,3])
        
    #     if name in ["Left"]:
    #         index.pop(0)
            
    #     elif name in ["Right"]:
    #         index.pop(0)
    #         temp_boundary_points = np.ones([size,3])
        
    #     elif name in ["Front"]:
    #         index.pop(1)

    #     elif name in ["Back"]:
    #         index.pop(1)
    #         temp_boundary_points = np.ones([size,3])
        
    #     elif name in ["Bottom"]:
    #         index.pop(2)

    #     else:
    #         index.pop(2)
    #         temp_boundary_points = np.ones([size,3])

        
    #     if len(index)==2:
            
    #         temp_boundary_points[:,index] = np.random.uniform(low=[0,0], high=[1,1], size = [size,2])
    #         BC_data = temp_boundary_points * (coord_max - coord_min) + coord_min
    #         np.save('./BC_%s_data.npy'%name,BC_data)
    #         data_config = ExistedDataConfig(name='BC_%s'%name,
    #                                         data_dir=['./BC_%s_data.npy'%name],
    #                                         columns_list=['data'], data_format="npy", constraint_type="Function")


    #     return data_config


    # #用Existed_data加载边界数据
    # existed_data_list = []

    # for name in BC_names:
    #     data_config = boundary_data_generate_func(name)
    #     existed_data_list.append(data_config)




    # create dataset for train and test
    train_dataset = Dataset(geometry_dict=geom_dict)

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
    # print(axis_x)
    axis_y = np.linspace(coord_min[1], coord_max[1], num=axis_size, endpoint=True)
    axis_z = np.linspace(coord_min[2], coord_max[2], num=axis_size, endpoint=True)
    mesh_x, mesh_y, mesh_z = np.meshgrid(axis_x, axis_y, axis_z,indexing='ij')
    input_data = np.hstack((mesh_x.flatten()[:, None], mesh_y.flatten()[:, None], mesh_z.flatten()[:, None])).astype(np.float32)

    # data = input_data.reshape(3,3,3,-1)

    # print(data[0,0,:,:]) #对应的是z轴这条线上的值
    # print(data[0,:,:,:]) #对应的是left这个面上的三维坐标
    # print(data[0,:,:,1:]) #对应的是left这个面上的y，z
    

    # data = []
    # for x in axis_x:
    #     for y in axis_y:
    #         for z in axis_z:
    #             data.append([x,y,z])
    
    # input_data = np.array(data)

    # print(input_data.reshape(3,3,3,-1))


    # input_data = np.hstack((mesh_x.flatten()[:, None], mesh_y.flatten()[:, None], mesh_z.flatten()[:, None])).astype(np.float32)
    # input_data = np.hstack((mesh_x[:, None], mesh_y.flatten()[:, None], mesh_z.flatten()[:, None])).astype(np.float32)
    # # label
    # label = np.zeros((axis_size, axis_size, 1))
    # for i in range(axis_size):
    #     for j in range(axis_size):
    #         label[i, j, 0] = np.sin(wave_number * axis_x[j])

    # label = label.reshape(-1, 1).astype(np.float32)

    return input_data #, label
