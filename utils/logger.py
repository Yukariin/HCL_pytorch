import os
import tqdm
import logging

from utils.dist import is_main_process, main_process_only


@main_process_only
def get_tb_writer(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    os.makedirs(log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir)
    return tb_writer


class StatusTracker:
    def __init__(self, logger, exp_dir: str, print_freq: int = 0):
        self.logger = logger
        self.print_freq = print_freq
        self.tb_writer = None
        if exp_dir is not None:
            self.tb_writer = get_tb_writer(log_dir=os.path.join(exp_dir, 'tensorboard'))

    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.close()

    def track_status(self, name: str, status: dict, step: int, write_tb: list[bool] = None):
        message = f'[{name}] step: {step}'
        for i, (k, v) in enumerate(status.items()):
            message += f', {k}: {v:.6f}'
            if self.tb_writer is not None and (write_tb is None or write_tb[i] is True):
                self.tb_writer.add_scalar(f'{name}/{k}', v, step)
        if self.print_freq > 0 and (step + 1) % self.print_freq == 0:
            self.logger.info(message)


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:  # noqa
            self.handleError(record)


def get_logger(name='exp', log_file=None, log_level=logging.INFO, file_mode='w', use_tqdm_handler: bool = False):
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger
    
    if not use_tqdm_handler:
        stream_handler = logging.StreamHandler()
    else:
        stream_handler = TqdmLoggingHandler()
    handlers = [stream_handler]

    if is_main_process() and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level if is_main_process() else logging.ERROR)
        logger.addHandler(handler)
    logger.setLevel(log_level if is_main_process() else logging.ERROR)
    
    return logger