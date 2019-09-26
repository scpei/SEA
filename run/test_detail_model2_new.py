from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import multiprocessing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import gc
import numpy as np
import time

from numpy import linalg as LA
import tensorflow as tf
import time
from tester_SEA2 import Tester

filename = 'en_fr'
model_file = './test-model_' + filename + '.ckpt'
data_file = 'test-multiG_' + filename + '.bin'
test_data = '../example_data/15k/en_fr_dict_15k_test.txt'

tester = Tester()
tester.build(save_path = model_file, data_save_path = data_file)
tester.load_test_data(test_data, splitter = '@@@@', line_end = '\n')

def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return []
    if n > ls_len:
        return []
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return

def cal_rank_multi_embed(frags, dic, sub_embed, embed, top_k):
    mean = 0
    mrr = 0
    num = np.array([0 for k in top_k])
    mean1 = 0
    mrr1 = 0
    num1 = np.array([0 for k in top_k])
    sim_mat = np.matmul(sub_embed, embed.T)
    prec_set = set()
    aligned_e = None
    for i in range(len(frags)):
        ref = frags[i]

        rank = (-sim_mat[i, :]).argsort()
        aligned_e = rank[0]
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
        # del rank

        if dic is not None and dic.get(ref, -1) > -1:
            e2 = dic.get(ref)
            sim_mat[i, e2] += 1.0
            rank = (-sim_mat[i, :]).argsort()
            aligned_e = rank[0]
            assert ref in rank
            rank_index = np.where(rank == ref)[0][0]
            mean1 += (rank_index + 1)
            mrr1 += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    num1[j] += 1
            # del rank
        else:
            mean1 += (rank_index + 1)
            mrr1 += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    num1[j] += 1

        prec_set.add((ref, aligned_e))

    del sim_mat
    gc.collect()
    return mean, mrr, num, mean1, mrr1, num1, prec_set

def eval_alignment_multi_embed(embed1, embed2, top_k, selected_pairs, mess=""):
    def pair2dic(pairs):
        if pairs is None or len(pairs) == 0:
            return None
        dic = dict()
        for i, j in pairs:
            if i not in dic.keys():
                dic[i] = j
        assert len(dic) == len(pairs)
        return dic

    t = time.time()
    dic = pair2dic(selected_pairs)
    ref_num = embed1.shape[0]
    t_num = np.array([0 for k in top_k])
    t_mean = 0
    t_mrr = 0
    t_num1 = np.array([0 for k in top_k])
    t_mean1 = 0
    t_mrr1 = 0
    t_prec_set = set()
    frags = div_list(np.array(range(ref_num)), 1)
    pool = multiprocessing.Pool(processes=len(frags))
    reses = list()
    for frag in frags:
        reses.append(pool.apply_async(cal_rank_multi_embed, (frag, dic, embed1[frag, :], embed2, top_k)))
    pool.close()
    pool.join()

    for res in reses:
        mean, mrr, num, mean1, mrr1, num1, prec_set = res.get()
        t_mean += mean
        t_mrr += mrr
        t_num += num
        t_mean1 += mean1
        t_mrr1 += mrr1
        t_num1 += num1
        t_prec_set |= prec_set

    assert len(t_prec_set) == ref_num

    acc = t_num / ref_num
    for i in range(len(acc)):
        acc[i] = round(acc[i], 4)
    t_mean /= ref_num
    t_mrr /= ref_num
    print("{}, hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(mess, top_k, acc, t_mean, t_mrr,
                                                                                 time.time() - t))
    if selected_pairs is not None and len(selected_pairs) > 0:
        acc1 = t_num1 / ref_num
        for i in range(len(acc1)):
            acc1[i] = round(acc1[i], 4)
        t_mean1 /= ref_num
        t_mrr1 /= ref_num
        print("{}, hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(mess, top_k, acc1, t_mean1, t_mrr1,
                                                                                     time.time() - t))
    return t_prec_set

index = 0
ent1 = []
ent2 = []
while index < len(tester.test_align):
    id = index
    index += 1
    e1, e2 = tester.test_align[id]
    #vec_e1 = tester.ent_index2vec(e1, source = 1)
    vec_proj_e1 = tester.projection(e1, source = 1)
    vec_e2 = tester.ent_index2vec(e2, source= 2)
    ent1.append(vec_proj_e1)
    ent2.append(vec_e2)

ent1 = np.asarray(ent1)
ent2 = np.asarray(ent2)

eval_alignment_multi_embed(ent1, ent2, [1, 5, 10], selected_pairs= None, mess="ent alignment")

