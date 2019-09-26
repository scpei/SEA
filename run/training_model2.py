from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from KG import KG
from multiG import multiG
from trainer2 import Trainer


filename = 'en_fr'
model_path = './test-model_' + filename + '.ckpt'
data_path = 'test-multiG_' + filename + '.bin'
kgf1 = '../example_data/15k/P_en_v5.csv'
kgf2 = '../example_data/15k/P_fr_v5.csv'
alignf = '../example_data/15k/en_fr_dict_15k.txt'
kgfreq1 = '../example_data/15k/freq_ef_en_15k.txt'
kgfreq2 = '../example_data/15k/freq_ef_fr_15k.txt'

this_dim = 100

KG1 = KG()
KG2 = KG()
KG1.load_triples(filename = kgf1, splitter = '@@@', line_end = '\n')
KG2.load_triples(filename = kgf2, splitter = '@@@', line_end = '\n')
KG1.load_freq(filename = kgfreq1, freq_split = 30, r_ = '!!')
KG2.load_freq(filename = kgfreq2, freq_split = 20, r_ = '!!')
this_data = multiG(KG1, KG2)
this_data.load_align(filename = alignf, lan1 = 'en', lan2 = 'fr', splitter = '@@@@', line_end = '\n')


m_train = Trainer()
m_train.build(this_data, dim=this_dim, batch_sizeK=1024, batch_sizeA=128, batch_sizeH = 128, a1=1.25, a2=0.5, m1=0.5, save_path = model_path, multiG_save_path = data_path, L1=False)
m_train.train_SEA( epochs=2000, save_every_epoch=50, lr=0.002, lr_ad=0.001, a1=1.25, a2=0.5, m1=0.5, AM_fold=5, half_loss_per_epoch=50)






