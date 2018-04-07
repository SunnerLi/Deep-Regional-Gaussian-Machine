Deep Regional Gaussian Machine
---
[![Packagist](https://img.shields.io/badge/Pytorch-0.3.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/Python-3.5.2-blue.svg)]()

![](https://i.imgur.com/6zyxWUD.jpg)

Abstraction
---
In this repository, we demonstrate another idea which called deep regional gaussian machine. Additionally, the regional gaussian module can be deployed to any common deep learning model. The deep regional gaussian machine is also tested toward MNIST and CIFAR10 dataset. The results show that the faster convergence phenomenon can be obtained. The detail is described in the [report](https://github.com/SunnerLi/Deep-Regional-Gaussian-Machine/blob/master/doc/deep%20regional%20gaussian%20machine.pdf). 

Usage
---
1. Move the `regionGaussian` folder and the containing to the same folder of your program    
2. Import the library:
```python
from regionGaussian.layer import RegionGaussian
```
3. Use it in the beginning of your model:
```python
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.rg = RegionGaussian(region_size = 1, adopt_concat = True)
        # Define the rest part of your model

    def forward(self, x):
        x = self.rg(x)
        # Define the rest forward process of your model
```

Result
---
The self-gaussian or region size = 3 can obtain the convergence benefit.    
![](https://github.com/SunnerLi/Deep-Regional-Gaussian-Machine/blob/master/img/experiment3.png)