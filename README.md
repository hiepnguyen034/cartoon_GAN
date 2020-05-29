This is a simple DCGAN network that generates cartoon faces. The code from this implementation is pretty much borrowed form this awesome PyTorch [tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) with some minor adjustments. The only purpose of this repo is to explore/experiment GAN and have some fun. 

While the dataset used in this repos is cartoon faces, the code can pretty much generate all kinds of images.

use `python run.py --begin_training True --img_path [PATH to training images directory]` to train model

use `python run.py --generate_img True --num_img [number of images]` to generate image(s) from random noise 

For more options, see `python run.py -h`

# Example:

Noise:

![img 1](https://user-images.githubusercontent.com/29159878/83233879-96d84900-a1b9-11ea-9dd7-f3bb5ea893f3.jpg)

Generated output:

![img2](https://user-images.githubusercontent.com/29159878/83233900-9b9cfd00-a1b9-11ea-94eb-886e7ea1fb83.jpg)

You should be able to build a grid of whatever size you configure.

An example of grid size =3, which means running (`python run.py --generate_img True --num_img 9`):

![single_output](https://user-images.githubusercontent.com/29159878/83252126-e8da9800-a1d4-11ea-9199-ecdcc943bc36.jpg)