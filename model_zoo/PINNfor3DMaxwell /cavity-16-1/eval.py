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
from mindspore import Tensor
from mindspore.common.initializer import HeUniform

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, ms_function
from mindspore.common import set_seed
from mindspore.train.callback import LossMonitor
from mindspore.train.loss_scale_manager import DynamicLossScaleManager

from mindelec.solver import Solver, Problem
from mindelec.geometry import HyperCube, create_config_from_edict
from mindelec.common import L2
from mindelec.data import Dataset
from mindelec.operators import SecondOrderGrad as Hessian
from mindelec.operators import Grad
from mindelec.loss import Constraints
from mindelec.data import ExistedDataConfig

from src.dataset import create_train_dataset
from src.loss import CustomWithLossCell
from src.logger import get_logger
from plot_seg import plot_cross_section, plot_waveguide,plot_slice
from src.config import  Maxwell_3d_config
from src.dataset import test_data_prepare
from src.callback import predict
from mindelec.architecture import MultiScaleFCCell
import json
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

    # input ????????????????????????
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

    # mesh_u, mesh_v, plane_0, plane_x, plane_y, plane_z = slice_data_prepare(Maxwell_3d_config)

    # # print(plane_0.reshape(51,51,3))
    # # test_input_data, test_prediction = predict(model=model,input_data=plane_0)
    # # print(test_prediction.reshape(51,51,3))
    # # print('xxxxxxxx')
    # _, test_prediction = predict(model=model,input_data=plane_x)
    # plot_slice(mesh_u, mesh_v,test_prediction,fig_fn='plane_x')

    # _, test_prediction = predict(model=model,input_data=plane_y)
    # plot_slice(mesh_u, mesh_v,test_prediction,fig_fn='plane_y')

    # _, test_prediction = predict(model=model,input_data=plane_z)
    # plot_slice(mesh_u, mesh_v,test_prediction,fig_fn='plane_z')

    # test_input = test_data_prepare(Maxwell_3d_config)
    # test_input_data, test_prediction = predict(model=model,input_data=test_input)
    # plot_cross_section(test_input_data,test_prediction,config=Maxwell_3d_config)

    # plot_waveguide(test_input_data,test_prediction,config=Maxwell_3d_config)

    data = np.load("Inf16.0.npz",allow_pickle=True)

    # print(data.files)

    val = data['arr_0'].item()

    print(val.keys())

    x = val['x']
    y = val['y']
    z = val['z']
    ux= val['ux']
    uy = val['uy']
    uz = val['uz']

    batch_size = Maxwell_3d_config["batch_size"] 


    domain_data = np.concatenate((x,y,z),axis=-1)[0:batch_size]
    domain_label = np.concatenate((ux,uy,uz),axis=-1)[0:batch_size]
    print(domain_data)
    # loss = net_with_loss(Tensor(domain_data,ms.float32),Tensor(plane_0,ms.float32))
    # print(loss)
    _, pre_domain_data = predict(model=model,input_data=domain_data)

    print(pre_domain_data)
    print(domain_label)
    print(pre_domain_data-domain_label)
    diff = pre_domain_data-domain_label
    print(np.mean(diff[:,0]**2))
    print(np.mean(diff[:,1]**2))
    print(np.mean(diff[:,2]**2))
            
        

if __name__ == '__main__':
    eval()
