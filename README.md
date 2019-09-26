Demo code of SEA.

The implementation is based on the code and data of MTransE which can be cited below. Thanks Dr.Muhao for his fundamental work and contributions.

This version is based on entity-level alignment, instead of triple-level alignment.

Contact: Shichao Pei (shichao.pei@kaust.edu.sa)

@inproceedings{chen2017multilingual,
  title={Multilingual knowledge graph embeddings for cross-lingual knowledge alignment},
  author={Chen, Muhao and Tian, Yingtao and Yang, Mohan and Zaniolo, Carlo},
  booktitle={Proceedings of the 26th International Joint Conference on Artificial Intelligence},
  pages={1511--1517},
  year={2017},
  organization={AAAI Press}
}

Usage:

To run the code, you need to have Python3 and Tensorflow installed.

Visit https://drive.google.com/file/d/1AsPPU4ka1Rc9u-XYMGWtvV65hF3egi0z/view to download the datasets.

The frequence of 20% or others need to be specified by manual, for example:
15000 records with frequence. you need to specify the frequence of 15000*(20%...) = 3000th entity in the whole record.
I have not implemented the function for automatic opertion. I will finish it in later.
