import sys

class Logger(object):
  def __init__(self, output_file):
    self.terminal = sys.stdout
    self.log = open(output_file, "a")

  def write(self, message):
    self.terminal.write(message)
    self.terminal.flush()
    self.log.write(message)
    self.log.flush()

  def flush(self):
  	self.terminal.flush()
  	self.log.flush()