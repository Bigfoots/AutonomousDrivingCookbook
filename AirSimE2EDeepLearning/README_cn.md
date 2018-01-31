# Autonomous Driving using End-to-End Deep Learning: an AirSim tutorial

### Authors:

**[Mitchell Spryn](https://www.linkedin.com/in/mitchell-spryn-57834545/)**, Software Engineer II, Microsoft

**[Aditya Sharma](https://www.linkedin.com/in/adityasharmacmu/)**, Program Manager, Microsoft

### Translator:  
**[Devin Yang](https://evyang1992.github.io/)**

## Overview

在本教程中，你将会学到如何用从[AirSim simulation environment](https://github.com/Microsoft/AirSim)  仿真环境搜集到的数据集来训练和测试用于自动驾驶的端对端深度学习模型。你的训练模型将会在 AirSim 的仿真地形中学会如何驾驶汽车，输入的视觉数据只来自一个设置在车前的摄像头。这套课程常被当作自动驾驶的入门项目，不过学完整套课程后， 你会有能力实现你自己的想法。  

这里是一个运行中模型的简短样本：  

![car-driving](car_driving.gif)



## 本文结构

教程里的代码都是在[Keras](https://keras.io/) 中实现的，Keras 是一种可以运行在[CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/)，[TensorFlow](https://www.tensorflow.org/) 或者[Theano](http://deeplearning.net/software/theano/index.html) 之上的深度学习 Python API。Keras 简单易用，是新手们的不二选择，能够削减大多数流行框架学习的难度。


这个教程会用 Python notebooks 的形式展现。Python notebooks 可以让你非常容易地阅读指导和说明，并且在一个文件中编写和运行代码，所有这一切都可以在浏览器窗口中完成。你可以按顺序浏览以下 notebooks：

**[DataExplorationAndPreparation](DataExplorationAndPreparation.ipynb)**

**[TrainModel](TrainModel.ipynb)**

**[TestModel](TestModel.ipynb)**

If you have never worked with Python notebooks before, we highly recommend [checking out the documentation](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html).
如果你之前从没有用过 Python notebooks，我们强烈推荐该教学文档：[checking out the documentation](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html)

## 依赖库和安装  

#### 背景依赖  

首先，你要了解基础的人工神经网络知识，不过高阶的概念就不需要了，比如 LSTM 或者强化学习。但是你应该要知道卷积网络的工作原理。Michael Nielsen 写的这本[神经网络和深度学习](http://neuralnetworksanddeeplearning.com/)非常不错，可在网上免费获取，它能够让你在一周之内构建坚实的神经网络知识基础。

同时，你还要会 Python，至少能够阅读和理解 Python 代码。

#### 依赖库

1. [安装 AirSim](https://github.com/Microsoft/AirSim#how-to-get-it)
2. [安装 Anaconda](https://conda.io/docs/user-guide/install/index.html) with Python 3.5 or higher.
3. [安装 CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine) or [install Tensorflow](https://www.tensorflow.org/install/install_windows)
4. [安装 h5py](http://docs.h5py.org/en/latest/build.html)
5. [安装 Keras](https://keras.io/#installation)
6. [配置 Keras backend](https://keras.io/backend/) TensorFlow (default) 或者 CNTK.

#### 硬件  

强烈建议用 GPU 来跑程序，虽说用 CPU 也能训练模型，但 CPU 要花数天才能完成训练。该教程用的是一块 GTX970 GPU，只需要 45 分钟就能完成训练。

如果你没有可用的 GPU，你可以用Azure[Deep Learning VM on Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.dsvm-deep-learning) 上的深度学习虚拟机，它安装之后会提供所有的依赖和库（此虚拟机需要 py35 环境）。

#### 数据集  

模型需要大量的数据集来训练，你可以在[这里]下载。第一个 notebook 会告诉你下载完成后如何获取这些数据。数据集最终解压后大小大概为 3.25 GB，虽说训练一辆真正的自动驾驶汽车需要 PB 级的数据，不过这些数据足够该教程的使用。

#### A note from the curators

We have made our best effort to ensure this tutorial can help you get started with the basics of autonomous driving and get you to the point where you can start exploring new ideas independently. We would love to hear your feedback on how we can improve and evolve this tutorial. We would also love to know what other tutorials we can provide you that will help you advance your career goals. Please feel free to use the GitHub issues section for all feedback. All feedback will be monitored closely. If you have ideas you would like to [collaborate](../README.md#contributing) on, please feel free to reach out to us and we will be happy to work with you.