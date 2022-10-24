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
train
"""
import os
import numpy as np
import math
import timeit

import mindspore as ms
from mindspore.common.initializer import HeUniform

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common import set_seed

from plot_seg import plot_cross_section, plot_waveguide
from src.config import  Maxwell_3d_config
from src.dataset import test_data_prepare
from src.callback import predict
from mindelec.architecture import MultiScaleFCCell
from mindspore.train.serialization import load_checkpoint, load_param_into_net

set_seed(0)
np.random.seed(0)

os.environ['CUDA_VISIBLE_DEVICES']='0'
print("pid:", os.getpid())

# prepare test input and label
def slice_data_prepare(config):
    """create test dataset"""
    coord_min = config["coord_min"]
    coord_max = config["coord_max"]
    axis_size = config["axis_size"]

    # input 认为是一个正方体
    u = np.linspace(coord_min[0], coord_max[0], num=axis_size, endpoint=True)
    v = np.linspace(coord_min[1], coord_max[1], num=axis_size, endpoint=True)
    mesh_u, mesh_v = np.meshgrid(u, v)
    
    
    uu,vv = mesh_u.flatten()[:, None], mesh_v.flatten()[:, None]
    ones =  np.ones((uu.shape[0],1))
    medium_value = (coord_max[2]+coord_min[2])/2.0
    
    plane_0 =  np.hstack(( coord_min[0] * ones, uu , vv)).astype(np.float32)
    plane_x =  np.hstack(( medium_value * ones, uu , vv)).astype(np.float32)
    plane_y =  np.hstack((  uu , medium_value * ones, vv)).astype(np.float32)
    plane_z =  np.hstack((  uu , vv, medium_value * ones)).astype(np.float32)

    return mesh_u, mesh_v, plane_0, plane_x, plane_y, plane_z


def eval():
    """eval process"""

    
    model =  MultiScaleFCCell(Maxwell_3d_config["input_size"],
                            Maxwell_3d_config["output_size"],
                            layers=Maxwell_3d_config["layers"],
                            neurons=Maxwell_3d_config["neurons"],     
                            weight_init=HeUniform(negative_slope=math.sqrt(5)),
                            act="sin",               
                             )

    # # # para1 = net.trainable_params()[0]
    # # # print(para1, "value:", para1.asnumpy())
    # # ms.save_checkpoint(net, "./MyNet_test.ckpt")

    # # param_dict = ms.load_checkpoint("MyNet_test.ckpt")
    # # ms.load_param_into_net(net,param_dict)

    # load parameters
    param_dict = load_checkpoint("maxwell3D.ckpt")
    convert_ckpt_dict = {}
    for _, param in model.parameters_and_names():
        convert_name1 = "model.cell_list." + param.name
        convert_name2 = "model.cell_list." + ".".join(param.name.split(".")[2:])
        for key in [convert_name1, convert_name2]:
            if key in param_dict:
                convert_ckpt_dict[param.name] = param_dict[key]
    load_param_into_net(model, convert_ckpt_dict)

    mesh_u, mesh_v, plane_0, plane_x, plane_y, plane_z = slice_data_prepare(Maxwell_3d_config)

    # print(plane_0.reshape(51,51,3))
    # test_input_data, test_prediction = predict(model=model,input_data=plane_0)
    # print(test_prediction.reshape(51,51,3))
    # print('xxxxxxxx')
    # _, test_prediction = predict(model=model,input_data=plane_x)
    # plot_slice(mesh_u, mesh_v,test_prediction,fig_fn='plane_x')

    # _, test_prediction = predict(model=model,input_data=plane_y)
    # plot_slice(mesh_u, mesh_v,test_prediction,fig_fn='plane_y')

    # _, test_prediction = predict(model=model,input_data=plane_z)
    # plot_slice(mesh_u, mesh_v,test_prediction,fig_fn='plane_z')

    test_input = test_data_prepare(Maxwell_3d_config)
    test_input_data, test_prediction = predict(model=model,input_data=test_input)
    plot_cross_section(test_input_data,test_prediction,config=Maxwell_3d_config)

    waveguide_data = np.load('waveguide_data.npy')
    waveguide_label = np.load('waveguide_label.npy')
    _, waveguide_predict = predict(model=model,input_data=waveguide_data)
    # print(waveguide_predict)

    plot_waveguide(waveguide_data,waveguide_label,waveguide_predict)
            
        

if __name__ == '__main__':
    eval()
