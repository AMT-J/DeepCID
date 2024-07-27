# DeepCID
Raman spectra contain abundant information from molecules but are difficult to analyze, especially for the mixtures. Deep-Learning-Based Components Identification for Raman Spectroscopy (DeepCID) has been proposed for the problem of components identification. Convolution Neural Network (CNN) models have been established to predict the presence of the components in the mixtures.

<div align="center">
<img src="https://raw.githubusercontent.com/xiaqiong/DeepCID/master/Flowchart_DeepCID.jpg" width=403 height=316 />
</div>

# Installation
## Env

```
conda create -n DPCID python=3.8  
```

## Install dependent packages

**1.Numpy**

```
pip install numpy==1.19.2
```

**2.Tensorflow**

```
pip install tensorflow-gpu==2.5.0  
```

**3.Scipy**

```
pip install scipy==1.6.0
```

**4.Matplotlib**

```
pip install matplotlib==3.3.4  
```

**5.Cuda**  

```
conda install -c conda-forge cudatoolkit=11.2  
conda install -c conda-forge cudnn=8.1.0 
```

# Download the model and run directly

Since the model exceeded the limit, we have uploaded all the models and the  information of mixtures to Release.

[Download](https://github.com/AMT-J/DeepCID/releases/tag/v1.0.0)

**1.Training your model**

Run the file 'one_component_model.py'.The corresponding example data have been uploaded to the folder named 'augmented data'.

**2.Predict mixture spectra data**

Run the file 'DeepCID.py'.An example mixture data have been uploaded at Baidu SkyDrive (named  'mixture.npy', 'label.npy' and 'namedata.csv').Download the model and these example data，DeepCID can be reload and predict easily.

# Paper
[Paper](https://pubs.rsc.org/en/content/articlehtml/2019/an/c8an02212g)

# Contact

Zhi-Min Zhang: zmzhang@csu.edu.cn


