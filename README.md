### Setup

This code follows almost exactly:
<https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>, just
minor changes to make it run for various params.


Linux, Conda:

```
conda create -n hmd-pytorch python=3
source activate hmd-pytorch
conda install pytorch-cpu torchvision-cpu -c pytorch
conda install pandas jupyter
```

```
wget https://download.pytorch.org/tutorial/hymenoptera_data.zip
unzip hymenoptera_data.zip
```

### Running

Look at the `train.py` main function; muck around with params, and then:

```
./train.py
```

