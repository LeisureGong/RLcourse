import argparse

from features.datautil import *
from features_generator_main import *
from logutil import logutil


def run(instance_tuples, feature_generator, outfile, old_feature_file=None):
    fw = open(outfile, 'w')
    fp = None
    if old_feature_file:
        fp = open(old_feature_file, 'r')
    for (authorid, paperid, label) in instance_tuples:
        feature_list = feature_generator.get_feature(authorid, paperid)
        if fp:
            last_feature_list = fp.readline().strip()
            fw.write("%s,%s" % (last_feature_list, ','.join([str(x) for x in feature_list])) + "\n")
        else:
            fw.write("%d\t%s" % (label, ','.join(str(x) for x in feature_list)) + "\n")
    fw.close()
    if fp:
        fp.close()
    print("label-2")


def main(train_csv, valid_csv, info_csv, train_outfile, valid_outfile, train_old_feature_file, valid_old_feature_file,
         generators_list_file):
    mylogger = logutil()
    mylogger.info("loading the datas...")
    data = Data(train_csv, valid_csv, info_csv)
    main_feature_generator = features_generator_main(data, generators_list_file)
    mylogger.info("---生成train数据的features---")
    run(data.train_tuples, main_feature_generator, train_outfile, train_old_feature_file)
    mylogger.info("---生成valid数据的features---")
    run(data.valid_tuples, main_feature_generator, valid_outfile, valid_old_feature_file)


def parser_json(setting_json):
    import json
    keys = json.loads(open(setting_json).read())
    return keys['train_csv'], keys['valid_csv'], keys['data_dir'], keys['train_feature_file'], keys[
        'valid_feature_file']


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Generate the feature files.")
    # parser.add_argument("--setting", dest="setting_json", required=True)
    # args = parser.parse_args()
    # train_csv, valid_csv, data_dir, train_outfile, valid_outfile = parser_json(args.setting_json)
    train_csv = 'data/train.csv'
    valid_csv = 'data/test.csv'
    data_dir = 'data/'
    train_outfile = 'output/features_file/train'
    valid_outfile = 'output/features_file/valid'
    old_train = None
    old_valid = None
    generators_list_file = "generators.list"
    main(train_csv, valid_csv, data_dir, train_outfile, valid_outfile, old_train, old_valid, generators_list_file)