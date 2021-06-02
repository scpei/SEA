## Semi-Supervised Entity Alignment via Knowledge Graph Embedding with Awareness of Degree Difference

The implementation is based on the code and data of MTransE.

This version is based on entity-level alignment, instead of triple-level alignment.

Contact: Shichao Pei (shichao.pei@kaust.edu.sa)

## Usage:

To run the code, you need to have Python3 and Tensorflow installed.

run `run_train_test.sh`

Visit https://drive.google.com/file/d/1AsPPU4ka1Rc9u-XYMGWtvV65hF3egi0z/view to download the datasets.

The frequence of 20% or others need to be specified by manual, for example:
15000 records with frequence. you need to specify the frequence of 15000*(20%...) = 3000th entity in the whole record.

## Dependencies
* Python>=3.5
* Tensorflow>=1.1.0
* numpy
* scipy
* multiprocessing
* pickle
* heapq

## Reference
Please refer to our paper. 

    @inproceedings{pei2019semi,
      title={Semi-supervised entity alignment via knowledge graph embedding with awareness of degree difference},
      author={Pei, Shichao and Yu, Lu and Hoehndorf, Robert and Zhang, Xiangliang},
      booktitle={The World Wide Web Conference},
      year={2019}
    }
