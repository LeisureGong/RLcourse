import sys

from features import *


class features_generator_main:
    def __init__(self, data, generator_conf):
        self.feature_generator = list()
        self.read_generator_conf(data, generator_conf)

    def read_generator_conf(self, data, generator_conf):
        for line in open(generator_conf):
            line = line.strip()

            if len(line) == 0 or line[0] == '#':
                continue
            try:
                temp = globals()[line]
            except Exception as e:
                print("create feature generator % s error!" % line)
                print(e)
                sys.exit(-1)
            print(line)
            print(temp)
            self.feature_generator.append(temp(data))

    def get_feature(self, authorid, paperid):
        feature_list = list()
        for feature_generator in self.feature_generator:
            feature_list += feature_generator.get_feature(authorid, paperid)
        return feature_list
