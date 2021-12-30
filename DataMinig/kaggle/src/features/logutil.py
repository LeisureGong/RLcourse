import logging
import sys


class logutil:
    def __init__(self, level='DEBUG'):
        self.logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s\t%(levelname)-8s\t%(message)s', '%Y/%b/%d %H:%M:%S', )
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        self.logger.setLevel(level)
        self.logger.addHandler(stream_handler)

    def info(self, st):
        self.logger.info(st)

    def debug(self, st):
        self.logger.debug(st)

    def error(self, st):
        self.logger.error(st)
