<h1 align="center">HyperCAN: Hypernetwork-driven deep parameterized constitutive models for metamaterials</h1>
<h4 align="center">
<a href="https://doi.org/10.1016/j.eml.2024.102243"><img alt="Static Badge" src="https://img.shields.io/badge/DOI-https%3A%2F%2Fdoi.org%2F10.1016%2Fj.eml.2024.102243-blue"></a>
<a href="https://zenodo.org/records/13947082"><img alt="Static Badge" src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.13947082-blue"></a>
</h4>
<div align="center">
  <span class="author-block">
    <a>Li Zheng</a><sup>1</sup>,</span>
  <span class="author-block">
    <a>Dennis M. Kochmann</a><sup>1</sup>, and</span>
    <span class="author-block">
    <a>Siddhant Kumar</a><sup>2</sup></span>  
</div>
<div align="center">
  <span class="author-block"><sup>1</sup>ETH Zurich, </span>
  <span class="author-block"><sup>2</sup>TU Delft</span>
</div>

$~$
<p align="center"><img src="MultiscaleFEM.png#gh-light-mode-only"\></p>

## Introduction

We introduce HyperCAN, a machine learning framework that utilizes hypernetworks to construct adaptable constitutive artificial neural networks for a wide range of beam-based metamaterials under finite deformations. For a detailed explortion of our approach and its applications,  please refer to our publication [HyperCAN: Hypernetwork-driven deep parameterized constitutive models for metamaterials](https://doi.org/10.1016/j.eml.2024.102243).

## Installation
### Dependencies
The framework was developed and tested on Python 3.10.4 using CUDA 12.0. You can install the required dependencies by running: 
 ```bash 
 pip install -r requirements.txt
 ```

### Setup
To conduct similar studies as those presented in the publication, start by cloning this repository via

```
git clone https://github.com/li-zhengz/HyperCAN.git
```
Next, download the data and model checkpoints provided in the [ETHZ Research Collection](https://doi.org/10.3929/ethz-b-000699994). Unzip the dataset in the `dataset` folder and the pre-trained model in the `model_checkpoint.zip`, as shown below. You can also build your own dataset for the given FEM simulationd data by running `python loadData.py`.
```
.
├── data
│   ├── dataset
│   │   ├── 6000_train_dataset.pt
│   │   ├── 6000_test_truss_dataset.pt
│   │   ├── 6000_test_load_dataset.pt
│   │   └── 6000_test_load_truss_dataset.pt
│   ├── FEM_data (optional)
│   │   ├── train
│   │   │   └── [...]
│   │   ├── test_truss
│   │   │   └── [...]
│   │   ├── test_load
│   │   │   └── [...]
│   │   └── test_load_truss
│   │       └── [...]
├── model_checkpoint
│   ├── icnn_checkpoint.pth
│   └── graph_checkpoint.pth 
```
### Training
Once you have the tranining data ready, use the following command to start the training process. The training parameters are specified in the `config.yaml` file. 
```python
python main.py
```
### Evaluation

 To evaluate the model, run `validation.py` to obtain the coefficient of determination ($R^2$ score) and the normalized root mean square error (NRMSE) metrics for the trained model across different datasets.

## Citation
If this code is useful for your research, please cite our [publication](https://doi.org/10.1016/j.eml.2024.102243).

```bibtex
@article{ZHENG2024,
title = {HyperCAN: Hypernetwork-driven deep parameterized constitutive models for metamaterials},
journal = {Extreme Mechanics Letters},
pages = {102243},
year = {2024},
issn = {2352-4316},
url = {https://www.sciencedirect.com/science/article/pii/S2352431624001238},
author = {Li Zheng and Dennis M. Kochmann and Siddhant Kumar},
}
```
## Author
This code is developed and maintained by [Li Zheng](https://scholar.google.com/citations?user=dLCJjh4AAAAJ&hl=en). 

For further information or inquiries, feel free to contact li.zheng@mavt.ethz.ch.