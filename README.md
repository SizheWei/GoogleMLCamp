# Portrait Segmentation using Tensorflow

This script removes the background from an input image. You can read more about segmentation [here](http://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb)

### Setup
- Download the model from the link below

  ```sh
  wget http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz
  wget http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz
  ```

- unzip the model via running the script:

  ```sh
  ./setup.sh
  ```

### Test

Go ahead and use the script as specified below, to execute fast but lower accuracy model:

```sh
python3 seg.py sample.jpg sample.png 
```

For better accuracy, albiet a slower approach, go ahead and try :
``` shell
python3 seg.py sample.jpg sample.png 1
```

### Dependencies
>	tensorflow, PIL

### Sample Result
Input: 
![alt text](https://github.com/SizheWei/GoogleMLCamp/blob/master/image_4.jpg "Input")

Output: 
![alt text](https://github.com/SizheWei/GoogleMLCamp/blob/master/output_4.png "Output")

