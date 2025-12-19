## Faster-RCNN implementation with ResNet50 Backbone 
The following code will be based on the Faster R-CNN with a ResNet50 backbone implementation. The code should be pretty easy to follow since I have provided extensive documentation using inline comments and through markdown cells. Any code that was borrowed by another source has been referenced using inline comments. Since this was a very intensive deep learning model the following code was built and trained locally using a Nvidia RTX 3070 GPU, so this expects most of the code to run locally, but you can potentially use a cloud tool such as Colab.

## Prerequistes
You will need the following pieces of software to run this locally:
- [Python](https://www.python.org/downloads/)
- [Pip](https://packaging.python.org/en/latest/tutorials/installing-packages/)
- If you have a Nvidia GPU you will want to install the [CUDA:12.8 toolkit](https://developer.nvidia.com/cuda-12-8-0-download-archive). After installing that you can go to the [pytorch website](https://pytorch.org/get-started/locally/) to install the cuda version of pytorch. You could also run the following command on Windows:
    ```bash
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    ```

## Directory Structure
```
Faster-RCNN/
├── README.md # You are here
├── R_CNNModel.ipynb # The main script for model
├── requirements.txt # List of libraries
├── models/ # Where saved models are stored
│   └── saved_models.pt
├── findings/ # All the results from runs
    └── findings.txt
```

## Installation and Usage
1. Create a virtual environment using the following command:
    ```bash
    python -m venv myenv
    ```
2. Activate the virtual environment using the following command:
    
    ```bash
    myenv\Scripts\activate
    ```
3. Now we can install all the packages we need using the following command:
    ```bash
    pip install -r requirements.txt
    ```
4. Open up a Jupyter Notebook or any text editor that can support `ipynb` files and start running the cells

## Configuration
Before we begin you will need to install the kaggle dataset onto your machine locally and then modify the `path` variable in the second cell to tell the script exactly where the dataset is stored. Additionally when using the `use_saved_model` function you will need to provide a path to the `.pt` file to use your saved pytorch model.