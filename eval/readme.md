# Evaluating corruptions
* Ensure that `test_dir` is pointing to the `dataset/test/` split
* Also ensure that the path to model checkpoints are set correctly
```
python eval_corruptions.py 
```
# Computing Central Kernel Alignment score for layers
* Ensure that you have at least 20 GB of GPU RAM to run the scripts
* Modify the path to save the heatmap visualization appropriately
```
python cka_convnext_mnasnet.py
```
