import pickle
import argparse


def parser_feature_select_str(st):
    cols = st.split('_')
    feature_list = list()
    for sub_str in cols:
        sub_begin_end = sub_str.split('-')
        if len(sub_begin_end) == 1:
            feature_list.append(int(sub_begin_end[0]) - 1)
        elif len(sub_begin_end) == 2:
            if sub_begin_end[1] != 'end':
                feature_list.append((int(sub_begin_end[0]) - 1, int(sub_begin_end[1]) - 1))
            else:
                feature_list.append((int(sub_begin_end[0]) - 1, 'end'))
    return feature_list


def load_ins_file(ins_file, feature_select_str):
    y = []
    x = []
    select_feature_list = parser_feature_select_str(feature_select_str)

    for line in open(ins_file):
        cols = line.strip().split()
        y.append(int(cols[0]))
        fea_values = map(float, cols[1].split(','))
        fea_values = list(cols[1].split(','))
        fea_value_list = list()
        for fea in select_feature_list:
            if type(fea) == int:
                fea_value_list.append(fea_values[fea])
            else:
                if fea[1] == 'end':
                    fea_value_list += fea_values[fea[0]:]
                else:
                    fea_value_list += fea_values[fea[0]:fea[1] + 1]
        x.append(fea_value_list)

    return x, y


def run_clf_main(valid_file, model_file, feature_select_str):
    valid_x, valid_y = load_ins_file(valid_file, feature_select_str)
    pkl_file = open(model_file, 'rb')
    clf = pickle.load(pkl_file)
    predictions = clf.predict_proba(valid_x)[:, 1]
    return predictions


def parser_json(setting_json):
    import json
    keys = json.loads(open(setting_json).read())
    return keys['valid_feature_file'], keys['model_path']


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='train the gbdt model')
    # parser.add_argument('--setting', dest='setting_json', required=True, help='the path of SETTINGS.json')
    # parser.add_argument('--cut_str', dest='cut_str', required=True, help='the cut str')
    # parser.add_argument('--learning_rate', dest='learning_rate', required=True, help='the learning rate of GBDT')
    #
    # args = parser.parse_args()
    #
    # valid_file, model_path = parser_json(args.setting_json)
    model_path = 'output/model/'
    trainfile = 'output/features_file/train'
    valid_file = 'output/features_file/valid'
    cut_str = '1-11_13-end'
    learning_rate = 0.8
    modelfile = '%s/GB-%s-%s.model.pickle' % (model_path, cut_str, learning_rate)
    output_prediction = '%s/GB-%s-%s.predictions' % (model_path, cut_str, learning_rate)
    # print valid_file, modelfile, output_prediction

    predictions = run_clf_main(valid_file, modelfile, cut_str)
    with open(output_prediction, 'w') as fp:
        for pred in predictions:
            fp.write(str(pred) + "\n" )
    output_real_name = '%s/GB-%s-%s.csv' % (model_path, cut_str, learning_rate)

    with open(output_real_name, "w") as f:
        f.write("Id,Expected\n")
        csv_id = 0
        for pred in predictions:
            expected = 0
            if pred > 0.5:
                expected = 1
            f.write(str(csv_id) + ',' + str(expected) + '\n')
            csv_id = csv_id + 1
