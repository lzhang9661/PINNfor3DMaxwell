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
feedforward neural network
"""

import mindspore.nn as nn
from mindelec.architecture import get_activation, LinearBlock
import numpy as np
from mindspore import Tensor
import mindspore as ms
import mindspore.ops as ops

class FFNN(nn.Cell):
    """
    Full-connect networks.

    Args:
    input_dim (int): the input dimensions.
    output_dim (int): the output dimensions.
    hidden_layer (int): number of hidden layers.
    activation (str or Cell): activation functions.
    """

    def __init__(self, input_dim, output_dim, hidden_layer=64, activation="sin"):
        super(FFNN, self).__init__()
        self.activation = get_activation(activation)
        mix_number=100
        self.mix_coe = Tensor(np.arange(1,mix_number+1)[np.newaxis,:],ms.float32)
        self.fc0 = LinearBlock(input_dim, mix_number,has_bias=False )
        self.fc1 = LinearBlock(mix_number, hidden_layer)
        self.fc2 = LinearBlock(hidden_layer, hidden_layer)
        self.fc3 = LinearBlock(hidden_layer, hidden_layer)
        self.fc4 = LinearBlock(hidden_layer, hidden_layer)
        self.fc5 = LinearBlock(hidden_layer, hidden_layer)
        self.fc6 = LinearBlock(hidden_layer, output_dim)
        self.print  = ops.Print()

        

    def construct(self, *inputs):
        """fc network"""
        x = inputs[0]
        # self.print(x.shape)
        out = self.fc0(x)*self.mix_coe
        # self.print(self.fc0(x).shape)
        # self.print(out.shape)
        out = self.activation(out)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        out = self.activation(out)
        out = self.fc4(out)
        out = self.activation(out)
        out = self.fc5(out)
        out = self.activation(out)
        out = self.fc6(out)
        return out


