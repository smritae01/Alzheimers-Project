# Assignment 3

## Prerequisites for running queries:

1. Python (3.7+)
2. [Pytest](https://docs.pytest.org/en/stable/)
3. Pytorch
4. Nibabel

## Input Data

Queries of Assignment 3 expect input files and directories as follows :

1. data directory : "data" consists of "datasets" and "output" directories 

(output directory is created when tasks are executed)

2. datasets directory : consists of "seg" directory, the npy files for all the MRIs, cnn_best.pth file and the ADNI3.csv metadata file.

(This is the directory where the bg_file and test_file are created during task 1)

3. output directory : will consist of SHAP directory and task-1.csv file.

4. SHAP directory : will consist of data and heatmaps directories.

5. SHAP/data directory : consists of the shap_value npy arrays from task 2.

6. SHAP/heatmaps directory : consists of the 2 heatmaps generated from task 3.

6. skeleton directory : consists of the data_util.py file, the explain_pipeline.py file, the model.py file and the Assignment3_test.py file.

## Running queries of Assignment 3

Example command :

python explain_pipeline.py --task 3 --dataFolder ../data/datasets/ADNI3  --outputFolder ../data

