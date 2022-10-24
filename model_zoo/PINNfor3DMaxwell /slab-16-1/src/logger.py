import logging
import os
import time
def get_logger( verbosity=1, name='None'):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])


    # 先在项目目录下建一个logs目录，来存放log文件
    logdir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'logs')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    # 再在logs目录下创建以日期开头的.log文件
    logfile = os.path.join(logdir,time.strftime('%Y-%m-%d') + '.log')

 
    fh = logging.FileHandler(logfile, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
 