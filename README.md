# SLRC
>  *Short-term and Life-time Repeat Consumption (SLRC)*     
>
> Model repeat consuming behavior with combination of Collaborative Filtering (CF) and Hawkes Process.



This is our implementation for the paper:

*Chenyang Wang, Min Zhang, Weizhi Ma, Yiqun Liu, and Shaoping Ma. 2019. [Modeling Item-Specific Temporal Dynamics of Repeat Consumption for Recommender Systems.](http://www.thuir.cn/group/~YQLiu/publications/WWW2019Wang.pdf) 
In WWW'19.*

Three models are implemented when integrating different CF methods:

- SLRC_BPR: [Bayesian Personalized Ranking (BPR)](https://dl.acm.org/citation.cfm?id=1795167) is a matrix factorization model with pairwise loss.
- SLRC_Tensor: [Tensor](https://dl.acm.org/citation.cfm?id=1864727) incorporates time factor with tensor factorization.
- SLRC_NCF: [Neural Collaborative Filtering (NCF)](https://dl.acm.org/citation.cfm?id=3052569) utilize nerual network to solve CF problem.

**Please cite our paper if you use our codes. Thanks!**

```
@inproceedings{wang2019modeling,
  title={Modeling Item-Specific Temporal Dynamics of Repeat Consumption for Recommender Systems},
  author={Wang, Chenyang and Zhang, Min and Ma, Weizhi and Liu, Yiqun and Ma, Shaoping},
  booktitle={The World Wide Web Conference},
  pages={1977--1987},
  year={2019},
  organization={ACM}
}
```

Author: Chenyang Wang (THUwangcy@gmail.com)



## Environments

```
python 3.5.2
tensorflow 1.2.0
pandas 0.23.4
numpy 1.16.0
sklearn 0.20.2
tqdm 4.29.1
```

If you want to train with GPU, tensorflow-gpu==1.2.0 is required.



## Datasets

- **Baby**: Online purchasing information within baby category from an e-commerce retailer. Not publicly available.
- **Order**: Mobile payment records in supermarkets and convenient stores. Not publicly available. But *we have release anonymized data [online](https://drive.google.com/drive/folders/1ZDjnC2L0pWpdqd5TNXMICaB_GOsZXTI8?usp=sharing)*.
- **Recsys2017**: Click log of job posting. The original data and description is avaliable [here](http://www.recsyschallenge.com/2017/).
- **BrightKite**: User check-in data to physical locations. The original data and description is avaliable [here](http://snap.stanford.edu/data/loc-brightkite.html).

The format description of origin data and dataset after preprocessing can be found in  [./data/README.md](https://github.com/THUwangcy/SLRC/tree/master/data).



## Usage		

Download [original data](https://drive.google.com/drive/folders/1ZDjnC2L0pWpdqd5TNXMICaB_GOsZXTI8?usp=sharing) and put it into ./data/ (or other data in the same format)

```
> cd SLRC/src
# make sure main.py can find necessary modules
> export PYTHONPATH=../

# generate dataset
> python Preprocess.py --dataset order

# SLRC_BPR in Order dataset
> python main.py --cf BPR --dataset order --gpu '' --K 100 --batch_size 256 --l2 1e-4 --lr 1e-4
```

According to our experiences, SLRC_BPR generally works well and takes less time to train, which is the most robust. SLRC_NCF relies on fine-tuned parameters to get good results (sometimes still worse than SLRC_BPR). Therefore, we *recommend to use BPR as the Collaborative filtering (CF) method* to calculate base intensity.

Example training log in Order dataset can be found in [./log/](https://github.com/THUwangcy/SLRC/tree/master/log).



Last Update Date: Jan 28, 2019
