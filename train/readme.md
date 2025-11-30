# Training scripts to train the following models.
   1. [Googlenet](https://arxiv.org/abs/1409.4842)
   2. [EfficientNet](https://arxiv.org/abs/1905.11946)
   3. [SqueezeNet](https://arxiv.org/abs/1602.07360)
   4. [RegNext](https://arxiv.org/abs/1611.05431v2)
   5. [Shufflenet](https://arxiv.org/abs/1707.01083)
   6. [MnasNet](https://arxiv.org/pdf/1807.11626)
   7. [Swin Transformer](https://arxiv.org/abs/2103.14030)
   8. [ConvNext](https://arxiv.org/abs/2201.03545)
   9. [VGG16](https://arxiv.org/abs/1409.1556)
 
* One could change the `build_model` function in any of the scripts to load an appropriate model.
* However, using a separate script for each model minimizes potential errors while studying the model performance.
* Hardware: A100 40 GB (one could also train the models in Google's Colaboratory)

# Running the script
* Assuming that the dataset directory is kept at one level above, and you want to train convnext
```
python convnext.py --data_dir ../dataset/ 
```
