# PyTorch EfficientDet API


A simple training, testing, and inference pipeline using [Ross Wightman's EfficientDet models](https://github.com/rwightman/efficientdet-pytorch). Ross Wightman's repo is used a submodule to load the EfficientDet models. 

The training/testing/inference code are custom written.

Get started with training within 5 minutes if you have the images and XML annotation files.



## Setup for Ubuntu

1. Clone the repository.

   ```
   git clone --recursive https://github.com/sovit-123/pytorch-efficientdet-api.git
   ```

2. Install requirements.

   1. **Method 1**: If you have CUDA and cuDNN set up already, do this in your environment of choice

      ```
      pip install -r requirments.txt
      ```

   2. **Method 2**: If you want to install PyTorch with CUDA Toolkit in your environment of choice.

      ```
      conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
      ```

      OR

      ```
      conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
      ```

      OR install the version with CUDA support as per your choice from **[here](https://pytorch.org/get-started/locally/)**.

      Then install the remaining **[requirements](https://github.com/sovit-123/pytorch-efficientdet-api/blob/main/requirements.txt)**.



## Setup on Windows

1. **First you need to install Microsoft Visual Studio from [here](https://my.visualstudio.com/Downloads?q=Visual%20Studio%202017)**. Sing In/Sing Up by clicking on **[this link](https://my.visualstudio.com/Downloads?q=Visual%20Studio%202017)** and download the **Visual Studio Community 2017** edition.

   ![](readme_images/vs-2017-annotated.jpg)

   Install with all the default chosen settings. It should be around 6 GB. Mainly, we need the C++ Build Tools.

2. Then install the proper **`pycocotools`** for Windows.

   ```
   pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI
   ```

3. Clone the repository.

   ```
   git clone --recursive https://github.com/sovit-123/pytorch-efficientdet-api.git
   ```

4. Install PyTorch with CUDA support.

   ```
   conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
   ```

   OR

   ```
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
   ```

   OR install the version with CUDA support as per your choice from **[here](https://pytorch.org/get-started/locally/)**.

   Then install the remaining **[requirements](https://github.com/sovit-123/pytorch-efficientdet-api/blob/main/requirements.txt)** except for `pycocotools`.
