# Beyond Freezing: Sparse Tuning Enhances Plasticity in Continual Learning with Pre-Trained Models

## 📝 Introduction
Continual Learning with Pre-trained Models holds great promise for efficient adaptation across sequential tasks. However, most existing approaches freeze PTMs and rely on auxiliary modules like prompts or adapters, limiting model plasticity and leading to suboptimal generalization when facing significant distribution shifts. While full fine-tuning can improve adaptability, it risks disrupting crucial pre-trained knowledge. In this paper, we propose Mutual Information-guided Sparse Tuning (MIST), a plug-and-play method that selectively updates a small subset of PTM parameters, less than 5\%, based on sensitivity to mutual information objectives. MIST enables effective task-specific adaptation while preserving generalization. To further reduce interference, we introduce strong sparsity regularization by randomly dropping gradients during tuning, resulting in fewer than 0.5\% of parameters being updated per step. Applied before standard freeze-based methods, MIST consistently boosts performance across diverse continual learning benchmarks. 
Experiments show that integrating our method into multiple baselines yields significant performance gains.
Our code will be available.


## 🔧 Requirements
###  Environment 
1. [torch 1.11.0](https://github.com/pytorch/pytorch)
2. [torchvision 0.12.0](https://github.com/pytorch/vision)
3. [timm 0.6.12](https://github.com/huggingface/pytorch-image-models)


### Dataset 
We provide the processed datasets as follows:
- **CIFAR100** : automatically downloaded by the provided code.

  ```bibtex
  @article{krizhevsky2009cifar,
    title={Learning multiple layers of features from tiny images},
    author={Krizhevsky, A. and Hinton, G.},
    journal={Handbook of Systemic Autoimmune Diseases},
    volume={1},
    number={4},
    year={2009},
  }
- **CUB200**:  Google Drive: [link](https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EVV4pT9VJ9pBrVs2x0lcwd0BlVQCtSrdbLVfhuajMry-lA?e=L6Wjsc) 
    ```bibtex  
    @article{wah2011cub,
    title={The caltech-ucsd birds-200-2011 dataset},
    author={Wah, Catherine and Branson, Steve and Welinder, Peter and Perona, Pietro and Belongie, Serge},
    year={2011},
    publisher={California Institute of Technology}
    }
- **ImageNet-R**: Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW)
  ```bibtex  
  @inproceedings{hendrycks2021many,
  title={The many faces of robustness: A critical analysis of out-of-distribution generalization},
  author={Hendrycks, Dan and Basart, Steven and Mu, Norman and Kadavath, Saurav and Wang, Frank and Dorundo, Evan and Desai, Rahul and Zhu, Tyler and Parajuli, Samyak and Guo, Mike and others},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={8340--8349},
  year={2021}
  }
- **ImageNet-A**:Google Drive: [link](https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/ERYi36eg9b1KkfEplgFTW3gBg1otwWwkQPSml0igWBC46A?e=NiTUkL)
    ```bibtex  
    @inproceedings{hendrycks2021natural,
    title={Natural adversarial examples},
    author={Hendrycks, Dan and Zhao, Kevin and Basart, Steven and Steinhardt, Jacob and Song, Dawn},
    booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
    pages={15262--15271},
    year={2021}
    }
- **Cars196**: Official link:  [link](https://tensorflow.google.cn/datasets/catalog/cars196) 
    ```bibtex  
    @inproceedings{krause2013cars,
    title={3d object representations for fine-grained categorization},
    author={Krause, Jonathan and Stark, Michael and Deng, Jia and Fei-Fei, Li},
    booktitle={Proceedings of the IEEE international conference on computer vision workshops},
    pages={554--561},
    year={2013}
    }
You need to modify the path of the datasets in `./utils/data.py`  according to your own path. 

## 💡 Running scripts
We provide implementations of MIST integrated into RanPac and SimpleCIL methods, available in `models/ranpac_mist.py` and `models/simplecil_mist.py`, respectively.

To prepare your JSON files, refer to the settings in the `exps` folder and run the following command. 

```
python main.py --config ./exps/[methodname]/[configname].json
```


## 🎈 Acknowledgement
This repo is based on https://github.com/zhoudw-zdw/RevisitingCIL and https://github.com/RanPAC/RanPAC
