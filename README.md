## Semi-Supervised Entity Alignment via Knowledge Graph Embedding with Awareness of Degree Difference

The study in this paper focuses on two important issues that limit the accuracy of current entity alignment solutions: 
1) labeled data of priorly aligned entity pairs are difficult and expensive to acquire, whereas abundant of unlabeled data are not used;
2) knowledge graph embedding is affected by entity’s degree difference, which brings challenges to align high frequent and low frequent entities.

The implementation is based on the code and data of MTransE.

This version is based on entity-level alignment, instead of triple-level alignment.

Contact: Shichao Pei (shichao.pei@kaust.edu.sa)

## Usage:

To run the code, you need to have Python3 and Tensorflow installed.

run `run_train_test.sh`

Visit https://drive.google.com/file/d/1AsPPU4ka1Rc9u-XYMGWtvV65hF3egi0z/view to download the datasets.

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
