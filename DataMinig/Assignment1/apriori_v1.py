import copy
import time
import pandas as pd


def apriori_joint(freq_sets, k):
    """
    将频繁k-1项通过拼接转换为候选k项集
    :param freq_sets:
    :param k:
    :return:
    """
    ck_frequent = []
    if k == 2:
        for i in range(len(freq_sets)):
            for j in range(i+1, len(freq_sets)):
                ck_frequent.append([freq_sets[i][0], freq_sets[j][0]])
    else:
        len_ck = len(freq_sets)
        for i in range(len_ck):
            # 取出k-2项元素
            curr_k_2 = list(freq_sets[i])[:k - 2]
            curr_k_2.sort()
            for j in range(i + 1, len_ck):
                next_k_2 = list(freq_sets[j])[k - 2]
                next_k_2.sort()
                if curr_k_2 == next_k_2:
                    ck_frequent.append(freq_sets[i] | freq_sets[j][-1])
    return ck_frequent

def preprocess(dataset, min_sup):
    """
    生成1-频繁项集
    :param dataset:
    :return:
    """
    c1_frequent_map = {}
    for item_set in dataset:
        for item in item_set:
            if item not in c1_frequent_map.keys():
                c1_frequent_map[item] = 1
            else:
                c1_frequent_map[item] += 1
    c1_frequent_resp = []
    for item_name in c1_frequent_map.keys():
        if c1_frequent_map[item_name] >= min_sup * len(c1_frequent_map.keys()):
            c1_frequent_resp.append([item_name])
    return c1_frequent_resp, min_sup * len(c1_frequent_map.keys())


# 计算候选集中每一项出现的频次
def candidate_frequency(data, candidate_k):
    freq_map = {}
    for item_set in candidate_k:
        can_set = set(item_set)
        can_str = ','.join(item_set)
        freq_map[can_str] = 0
        for init_data in data:
            init_set = set(init_data)
            if can_set.issubset(init_set):
                freq_map[can_str] += 1
    return freq_map


if __name__ == "__main__":
    path = "dataset/GroceryStore/Groceries.csv"
    source_data = pd.read_csv(path)
    dataset_list = source_data.values.tolist()
    dataset = list(map(lambda x: x[1].replace('{', '').replace('}', '').split(","), dataset_list))
    # 支持率
    min_support = 0.1
    # 置信度
    min_confidence = 0.5
    # 最大频繁项集大小
    max_itemset_size = 2

    ck_frequent, frequent_set_size = preprocess(dataset, min_support)
    for k in range(2, max_itemset_size + 1):
        start_time = time.time()
        candidate_k = apriori_joint(ck_frequent, k)
        candidate_k_map = candidate_frequency(dataset, candidate_k)
        ck_frequent = []
        new_candidate_k_map = {}
        for key in candidate_k_map.keys():
            if candidate_k_map[key] >= frequent_set_size:
                ck_frequent.append(sorted(key.split(',')))
                new_candidate_k_map[','.join(ck_frequent[-1])] = candidate_k_map[key]
        candidate_k_map = new_candidate_k_map
        if len(ck_frequent) == 0:
            break
        ck_frequent.sort()
        print(ck_frequent, candidate_k_map, frequent_set_size, k)
        print()
