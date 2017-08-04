import logging
import tensorboard_logger 
from collections import defaultdict
import sys
import math

class DeepSurvLogger():
    def __init__(self, name):
        self.logger         = logging.getLogger(name)
        self.history = {}

    def logMessage(self,message):
        self.logger.info(message)

    def print_progress_bar(self, step, max_steps, loss = None, ci = None, bar_length = 25, char = '*'):
        progress_length = int(bar_length * step / max_steps)
        progress_bar = [char] * (progress_length) + [' '] * (bar_length - progress_length)
        space_padding = int(math.log10(max_steps))
        if step > 0:
            space_padding -= int(math.log10(step))
        space_padding = ''.join([' '] * space_padding)
        message = "Training step %d/%d %s|" % (step, max_steps, space_padding) + ''.join(progress_bar) + "|"
        if loss:
            message += " - loss: %.4f" % loss
        if ci:
            message += " - ci: %.4f" % ci

        self.logger.info(message)

    def logValue(self, key, value, step):
        pass

    def shutdown(self):
        logging.shutdown()

class TensorboardLogger(DeepSurvLogger):
    def __init__(self, name, logdir, max_steps = None, update_freq = 10):
        self.max_steps = max_steps

        self.logger         = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        format = logging.Formatter("%(asctime)s - %(message)s")
        ch.setFormatter(format)
        self.logger.addHandler(ch)

        self.update_freq    = update_freq

        self.tb_logger = tensorboard_logger.Logger(logdir)

        self.history = defaultdict(list)

    def logValue(self, key, value, step):
        self.tb_logger.log_value(key, value, step)
        self.history[key].append((step, value))