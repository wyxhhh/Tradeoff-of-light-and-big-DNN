# Trade-off Between Energy Consuming and Accuracy

 We propose a machine learning based determiner that receives the outcomes of classification layer of small networks combined with data set size and complexity of small network in order to make the determiner generalized. Gradient-Boost Decision Tree (GBDT), Random Forest and Decision Tree are considered in determiner selection. It achieves 98.19% accuracy with penalty on calling the big network (i.e.,Acc − \lambda f), which is1.21% better than the traditional method.

## Configurations for Networks
| Name           | LeNet                       | Mini1(m1)                   | Mini2(m2)                   | Mini3(m3)                  | MIni4(m4) |
|----------------|-----------------------------|-----------------------------|-----------------------------|----------------------------|----------------------------|
| Input          | 28 × 28                | 28 × 28                | 28 × 28                | 28 × 28               | 28 × 28               |
|                | Conv2d 5 × 5 × 20 | Conv2d 5 × 5 × 10 | Conv2d 5 × 5 × 7  | Conv2d 5 × 5 × 2 | Conv2d 2 × 2 × 2 |
|                | MaxPool2d 2 × 2        | MaxPool2d 2 × 2        | MaxPool2d 2 × 2        | MaxPool2d 2 × 2       | MaxPool2d 2 × 2       |
|                | Conv2d 5 × 5 × 50 | Conv2d 5 × 5 × 25 | Conv2d 5 × 5 × 15 | Conv2d 5 × 5 × 5 | Conv2d 2 × 2 × 5 |
|                | MaxPool2d 2 × 2        | MaxPool2d 2 × 2        | MaxPool2d 2 × 2        | MaxPool2d 2 × 2       | MaxPool2d 2 × 2       |
|                | Linear 500                  | Linear 250                  | Linear 150                  | Linear 50                  | Linear 50                  |
|                | Linear 84                   | Linear 10                   | Linear 10                   | Linear 10                  | Linear 10                  |
|                | Linear 10                   |                             |                             |                            |                            |
| Size(MB)       | 2.05                        | 0.55                        | 0.25                        | 0.05                       | 0.08                       |
| Top-1 Accruacy | 99.16                       | 98.86                       | 98.73                       | 97.99                      | 97.81                      |


| Name           | Mini5(m5)                   | Mini6(m6)                   | Mini7(m7)                             | Mini8(m8)                             |
|----------------|-----------------------------|-----------------------------|---------------------------------------|---------------------------------------|
| Input          | 28 × 28                | 28 × 28                | 28 × 28                          | 28 × 28                          |
|                | Conv2d 5 × 5 × 20 | Conv2d 5 × 5 × 10 | Conv2d 5 × 5 × 10, stride 2 | Conv2d 5 × 5 × 10, stride 2 |
|                | MaxPool2d 2 × 2        | MaxPool2d 2 × 2        | MaxPool2d 2 × 2                  | MaxPool2d 2 × 2                  |
|                | Linear 500                  | Linear 250                  | Linear 250                            | Linear 250                            |
|                | Linear 10                   | Linear 10                   | Linear 10                             | Linear 10                             |
| Size(MB)       | 1.72                        | 1.49                        | 0.39                                  | 0.11                                  |
| Top-1 Accruacy | 98.77                       | 97.99                       | 98.05                                 | 95.18                                 |


## Data and Pre-preprocessing
We carried out experiments on MNIST. The data should be in the ```data``` folder. When considering the train data for the determiners, simply run the code below:
```shell
python data_preprocessing.py
```
We derive the origin input data for determiners training. Before training process, we need to do the normalization, to implement, run the code below:
```shell
python preprocessing.py
```
Or, get the input data from [here](https://jbox.sjtu.edu.cn/l/L04d4B)

## Train
### Train the Networks
Run the command below to train the big/small networks, you can train the network with specific data size by modifying the ```train_net``` python file.
```shell
python train_net.py
```
or just download the pre-trained models below:
- Download [mini1-2](https://jbox.sjtu.edu.cn/l/KnHzFd)
- Download [mini3-5](https://jbox.sjtu.edu.cn/l/noXqmh)
- Download [mini6-8](https://jbox.sjtu.edu.cn/l/Y0TMhn)

### Train the Determiners
To train and select the determiner with the best performance, run the code below:
```shell
python train_xxxx.py
```
You can choose ```DecisionTree``` or ```RandomForest``` to do the training.
Or, download the pre-trained [models](https://jbox.sjtu.edu.cn/l/fJ6wEP)