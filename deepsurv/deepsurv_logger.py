import logging
import tensorboard_logger 
from collections import defaultdict

class DeepSurvLogger():
	def __init__(self):
		pass

	def logMessage(self,message):
		self.logger.info(message)

	def print_progress_bar(self, step, max_steps, loss = None, bar_length = 25, char = '*', ):
		progress_length = int(bar_length * step / max_steps)
		progress_bar = [char] * (progress_length) + [' '] * (bar_length - progress_length)
		message = "Training step %d/%d |" % (step, max_steps) + ''.join(progress_bar) + "|"
		if loss:
			message += " - loss: %.4f" % loss
		self.logger.info(message)


class TensorboardLogger(DeepSurvLogger):
	def __init__(self, name, logdir, max_steps = None, update_freq = 10):
		self.max_steps = max_steps

		self.logger 		= logging.getLogger(name)
		self.update_freq 	= update_freq

		self.tb_logger = tensorboard_logger.Logger(logdir)

		self.history = defaultdict(list)

	def logValue(self, key, value, step):
		self.tb_logger.log_value(key, value, step)
		self.history[key].append((step, value))

		if self.max_steps and step % self.update_freq == 0:
			self.print_progress_bar(step, max_steps)

	