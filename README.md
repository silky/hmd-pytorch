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
# Grab flower dataset
!curl -LO http://download.tensorflow.org/example_images/flower_photos.tgz && \
  tar xzf flower_photos.tgz

# Remove license so it doesn't get picked up as a class.
!mv flower_photos/LICENSE.txt .


# Pick out 70 images for our hold-out validation set.
!for f in flower_photos/*; \
  do mkdir -p test/$f && find $f -type f | shuf -n 70 | xargs -I {} mv {} test/$f; \
done;

# Call the remaining photos our training ones,
# and rename the test ones so it all matches.
!mv flower_photos train
!mv test/flower_photos val
!rm -rf test
```

### Running

Look at the `train.py` main function; muck around with params, and then:

```
./train.py
```


### Todo

- Try it with _no_ retraining at all; just swapping out the last layer.
