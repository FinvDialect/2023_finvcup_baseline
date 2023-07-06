# 第八届信也科技杯baseline

数智创新，声至未来
Deep in Dialects, for Future Wave.

这是第八届信也科技的baseline。 
本届大赛以“智能语音质检，提升用户体验”为背景，探索利用AI技术识别和还原语音数据中的方言信息，特别是不同方言之间的距离特征的问题。这一问题有助于更好地理解汉语语音及其方言、口音特征，以及将相关技术从理论到实际应用的实现，以进一步支持对用户的更好服务。



## Environments
Implementing environment:  
- python=3.6
- torch==1.9.0
- torchaudio==0.9.0
- pydub==0.21.0
- numba==0.48.0
- numpy==1.15.4
- pandas==0.23.3
- scipy==1.2.1
- scikit-learn==0.19.1
- tqdm
- SoundFile==0.12.1

- GPU: Tesla V100 32G  



## Dataset
./data 目录下有所需的test_pair数据文件。

test_pair 包含提交所需的100万个数据对，需要选手提交对应的一百万个方言距离，并严格按照test_pair内的样本顺序 
  
音频数据请从共享地址下载：  
 
请将下载好的数据文件 train.zip 和 test.zip 置于工程根目录下，执行  
```bash
unzip "*.zip" -d ./data/
python create_data_index.py
```
解压文件并生成目录索引。文件索引选手可根据个人需求自行生成。

## Training


```bash
python train.py --loss aamsoftmax --max_epoch 80 --device cuda:0 --save_path ./exps/
```


```bash
python train.py --loss StandardSimilarityLoss --max_epoch 80 --device cuda:0 --save_path ./exps_sim/
```


```bash
python train.py --loss PairDistanceLoss --max_epoch 80 --device cuda:0 --save_path ./exps_pairdist/
```


## Inference
```bash
python inference.py --model_path exps/model/model_0001.model --test_path data/test --device cuda:0
```
会在根目录下生成提交所需的submit.csv文件

## Acknowledge
- We borrowed a lot of code from [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN) for modeling
