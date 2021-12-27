# MMDNE
Source code for CIKM 2019 paper ["**Temporal Network Embedding with Micro- and Macro-dynamics**"](https://arxiv.org/abs/1909.04246).

# Requirements

- Python 2.7
- numpy
- scipy
- PyTorch (0.3.0)
- My machine with two GPUs (NVIDIA GTX-1080 *2) and two CPUs (Intel Xeon E5-2690 * 2)

# Description

The datasets are also available at [Google Drive](https://drive.google.com/drive/folders/1Al9CfTn3BEu-dmE0dqMJfKwGOzY4frMc?usp=sharing).


```
MMDNE/
├── code
│   ├── DataHelper.py: load and process data for MMDNE
│   ├── Evaluation.py: evaluate the performance of MMDNE (e.g., classification)
│   └── MMDNE.py: model architecture and training
├── data
│   └── dblp
│       ├── dblp.txt: each line is a temporal edge with the format (node1 \t node2 \t timestamp)
│       ├── node2label.txt: node label data with the format (node_name, label)
│   └── Tmall
│       ├── tmall.txt: each line is a temporal edge with the format (node1 \t node2 \t timestamp)
│       ├── node2label.txt: node label data with the format (node_name, label)
│   └── Eucore: will be available soon!
└── res
│    └── dblp
│        └──
├── README.md
```

# Usage:
python MMDNE.py


# Reference

```
@inproceedings{Yuanfu2019MMDNE,
  title={Temporal Network Embedding with Micro- and Macro-dynamics},
  author={Yuanfu Lu, Xiao Wang, Chuan Shi, Philip S. Yu, Yanfang Ye.}
  booktitle={Proceedings of CIKM},
  year={2019}
}

```


