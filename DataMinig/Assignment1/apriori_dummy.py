import copy
import time
import pandas as pd

# 将频繁k-1项通过拼接转换为候选k项集
def apriori_joint(source_sets, k):
    ck_frequent = []
    len_ck = len(source_sets)
    for i in range(len_ck):
        # 取出k-2项元素
        curr_k_2 = list(source_sets[i])[:k - 2]
        curr_k_2.sort()
        for j in range(i + 1, len_ck):
            next_k_2 = list(source_sets[j])[k - 2]
            next_k_2.sort()
            if curr_k_2 == next_k_2:
                ck_frequent.append(source_sets[i] | source_sets[j][-1])
    return ck_frequent

# 由k-1频繁项集生成k候选项集
def apriori_gen_brute(freq_sets):
    # 找出k-1频繁项中包含的所有item
    items = []
    for item_set in freq_sets:
        for item in item_set:
            items.append(item)
    items = list(set(items))
    candidates = []
    for item_set in freq_sets:
        for item in items:
            if item not in item_set:
                itemset_bak = copy.deepcopy(item_set)
                itemset_bak.append(item)
                candidates.append(','.join(sorted(itemset_bak)))
    # 去重
    candidates_resp = []
    candidates = set(candidates)
    for item in candidates:
        candidates_resp.append(item.split(','))
    return candidates_resp

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
        if c1_frequent_map[item_name] >= min_sup*len(c1_frequent_map.keys()):
            c1_frequent_resp.append([item_name])
    return c1_frequent_resp, min_sup*len(c1_frequent_map.keys())


# 计算候选集中每一项出现的频次
def candidate_frequency(data, candidate_k):
    freq_map = {}
    for item_set in candidate_k:
        can_set = set(item_set)
        can_str = ','.join(item_set)
        freq_map[can_str] = 0
        for init_tup in data:
            if can_set.issubset(init_tup):
                freq_map[can_str] += 1
    return freq_map


if __name__ == "__main__":
    path = "dataset/GroceryStore/Groceries.csv"
    source_data = pd.read_csv(path)
    dataset_list = source_data.values.tolist()
    dataset_tmp = list(map(lambda x: x[1].replace('{', '').replace('}', '').split(","), dataset_list))
    dataset = [tuple(row) for row in dataset_tmp]
    # 支持率
    min_support = 0.01
    # 置信度
    min_confidence = 0.5
    # 最大频繁项集大小
    max_itemset_size = 5

    ck_frequent, frequent_set_size = preprocess(dataset, min_support)
    for k in range(2, max_itemset_size + 1):
        start_time = time.time()
        # candidate_k = apriori_joint(ck_frequent, k)
        candidate_k = apriori_gen_brute(ck_frequent)
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
