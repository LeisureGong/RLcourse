from sklearn.ensemble import *
import argparse

def load_ins_file(ins_file, feature_select_str):
    y = []
    x = []
    select_feature_list = parser_feature_select_str(feature_select_str)
    ins_file_data = open(ins_file, 'r')
    ins_file = ins_file_data.read()
    for line in ins_file:
        cols = line.strip().split()
        y.append(int(cols[0]))
        fea_values = map(float, cols[1].split(','))
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


def get_classifier(lr):
    return GradientBoostingClassifier(learning_rate=lr, n_estimators=500, min_samples_split=20, random_state=1)

def run_clf_main(train_file, model_file, cut_str, learning_rate):
    train_x, train_y = load_ins_file(train_file, learning_rate)


def parser_json(setting_json):
    import json
    keys = json.loads(open(setting_json).read())
    return keys['train_feature_file'], keys['model_path']

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="gbdt")
    parser.add_argument('--setting', dest='setting_json', required=True)
    parser.add_argument('--cut_str',dest='cut_str',required=True)
    parser.add_argument('--learning_rate',dest='learning_rate',required=True)

    args = parser.parse_args()
