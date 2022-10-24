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

import mindspore as ms
from mindspore.common.initializer import HeUniform

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.common import set_seed

from src.dataset import create_train_dataset
from src.loss import CustomWithLossCell
from src.logger import get_logger
from src.config import  Maxwell_3d_config
from src.dataset import test_data_prepare
from src.callback import TimeMonitor, Custom_LossMonitor,SaveCkptMonitor
from mindelec.architecture import MultiScaleFCCell

set_seed(0)
np.random.seed(0)

os.environ['CUDA_VISIBLE_DEVICES']='1'
print("pid:", os.getpid())
context.set_context(mode=context.GRAPH_MODE, save_graphs=False)

def train():
    """train process"""
    # net = FFNN(Maxwell_3d_config["input_size"],
    #            Maxwell_3d_config["output_size"], 
    #              Maxwell_3d_config["neurons"])
    
    train_dataset = create_train_dataset(Maxwell_3d_config)

    train_loader = train_dataset.create_dataset(batch_size=Maxwell_3d_config["batch_size"],
                                                shuffle=True, drop_remainder=True)
    
    net =  MultiScaleFCCell(Maxwell_3d_config["input_size"],
                            Maxwell_3d_config["output_size"],
                            layers=Maxwell_3d_config["layers"],
                            neurons=Maxwell_3d_config["neurons"],     
                            weight_init=HeUniform(negative_slope=math.sqrt(5)),
                            act="sin",               
                             )
    net.to_float(ms.float16)
    
    test_input = test_data_prepare(Maxwell_3d_config)

        
    # 连接前向网络与损失函数
    net_with_loss = CustomWithLossCell(net,Maxwell_3d_config)
    # optimizer
    optim = nn.Adam(net.trainable_params(), learning_rate=Maxwell_3d_config["lr"])

    
    # 使用Model连接网络和优化器，此时Model内部不经过nn.WithLossCell
    model = ms.Model(network=net_with_loss, optimizer=optim)

    logger = get_logger()


    SaveCkpt_cb =  SaveCkptMonitor(logger)
    time_cb = TimeMonitor(logger)
    loss_cb = Custom_LossMonitor(logger)
    # 使用train接口进行模型训练
    model.train(epoch=Maxwell_3d_config["Epochs"], train_dataset=train_loader, callbacks=[loss_cb,SaveCkpt_cb,time_cb]) #
    
  
if __name__ == '__main__':
    train()
