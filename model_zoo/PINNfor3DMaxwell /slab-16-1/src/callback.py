import mindspore.nn as nn
import mindspore as ms
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops

# from mindspore.train.callback import LossMonitor
from mindspore.train.callback._callback import Callback, _handle_loss

def predict(model,input_data):
    test_input_data = input_data.reshape(-1, 3)
    index = 0
    prediction = np.zeros([test_input_data.shape[0],3])
    while index < len(test_input_data):
        index_end = min(index + 8000, len(test_input_data))
        test_batch = Tensor(test_input_data[index: index_end, :], dtype=ms.float32)
        predict = model(test_batch)  #?这里应该输出的是pinn loss
        predict = predict.asnumpy()
        prediction[index: index_end, :] = predict[:, :]
        index = index_end
    return test_input_data,prediction



from mindspore.train.callback import Callback
from mindspore import Tensor
import mindspore.common.dtype as mstype

import time
class TimeMonitor(Callback):
    """
    Monitor the time in training.

    Args:
    data_size (int): Iteration steps to run one epoch of the whole dataset.
    """
    def __init__(self, logger,data_size=None):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size
        self.epoch_time = time.time()
        self.per_step_time = 0
        self.logger = logger

    def epoch_begin(self, run_context):
        """
        Set begin time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the train running.
        """
        run_context.original_args()
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """
        Print process cost time at the end of epoch.
        """
        epoch_seconds = (time.time() - self.epoch_time) * 1000
        step_size = self.data_size
        cb_params = run_context.original_args()
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num

        self.per_step_time = epoch_seconds / step_size
        self.logger.info("epoch time: {:5.3f} ms\t per step time: {:5.3f} ms \t".format(epoch_seconds, self.per_step_time))

    def get_step_time(self):
        return self.per_step_time


class Custom_LossMonitor(Callback):
    
    def __init__(self, logger):
        super(Custom_LossMonitor, self).__init__()
        
        self.logger = logger

    
    def on_train_epoch_end(self, run_context):
        """
        Print process Loss at the end of epoch.
        """
        cb_params = run_context.original_args()
        loss = _handle_loss(cb_params.net_outputs)

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1


        self.logger.info("epoch: %d, step: %s, loss is %s" % (cb_params.cur_epoch_num,cur_step_in_epoch, loss))



# 定义保存ckpt文件的回调接口
class SaveCkptMonitor(Callback):
    """定义初始化过程"""

    def __init__(self, logger,loss_thresh=1000,model_name="maxwell3D"):
        super(SaveCkptMonitor, self).__init__()
        self.loss = loss_thresh  # 定义损失值阈值
        self.logger = logger
        self.model_name = model_name


    def on_train_epoch_end(self, run_context):
        """定义每一个epoch结束时的执行操作"""
        cb_params = run_context.original_args()
        cur_loss = cb_params.net_outputs.asnumpy() # 获取当前损失值
        

        # 当loss比原来小才更新模型
        if cur_loss < self.loss:

            self.loss = cur_loss
            # 自定义保存文件名
            file_name = self.model_name  + ".ckpt"
            # 保存网络模型
            ms.save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
            self.logger.info("Saved checkpoint, loss:{:8.7f}, current epoch num:{:4}.".format(cur_loss, cb_params.cur_epoch_num))


