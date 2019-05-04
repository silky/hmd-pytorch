### Setup

This code follows almost exactly:
<https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>, just
minor changes to make it run for various params, and switched the dataset.

Best to run it on the Google CoLaboratory:

- <https://colab.research.google.com/drive/1VmYwWHKCnLnOA6snmLpzMLssBKvVKKab>

If you want to run it locally, run something like:

```
conda create -n hmd-pytorch python=3
source activate hmd-pytorch
conda install pytorch-cpu torchvision-cpu -c pytorch
conda install pandas jupyter
```

Then, get the data and put it in the right folder structure with:

```
echo Remove license so it doesn't get picked up as a class.
mv flower_photos/LICENSE.txt .


echo Pick out 70 images for our hold-out validation set.
for f in flower_photos/*; \
  do mkdir -p test/$f && find $f -type f | shuf -n 70 | xargs -I {} mv {} test/$f; \
done;

echo Call the remaining photos our training ones,
echo and rename the test ones so it all matches.
mv flower_photos train
mv test/flower_photos val
rm -rf test
```

Run just call the `go()` function!
