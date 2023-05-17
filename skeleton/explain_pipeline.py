import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import nibabel as nib
import numpy as np
import os
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import _CNN
import logging
from data_util import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
# python explain_pipeline.py --task 1 --dataFolder ../data/datasets/ADNI3  --outputFolder ../data

# This is a color map that you can use to plot the SHAP heatmap on the input MRI
colors = []
for l in np.linspace(1, 0, 100):
    colors.append((30./255, 136./255, 229./255,l))
for l in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255,l))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)


# Returns two data loaders (objects of the class: torch.utils.data.DataLoader) that are
# used to load the background and test datasets.
def prepare_dataloaders(bg_csv, test_csv, bg_batch_size = 8, test_batch_size= 1, num_workers=1):
    '''
    Attributes:
        bg_csv (str): The path to the background CSV file.
        test_csv (str): The path to the test data CSV file.
        bg_batch_size (int): The batch size of the background data loader
        test_batch_size (int): The batch size of the test data loader
        num_workers (int): The number of sub-processes to use for dataloader
    '''
    # YOUR CODE HERE
    train_data = CNN_Data(bg_csv)
    test_data = CNN_Data(test_csv)
    
    
    train_dataloader = DataLoader(train_data, bg_batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, test_batch_size, num_workers=num_workers)
    logger.info(f"train-dataloader size : {train_dataloader}")
    logger.info(f"test-dataloader size : {test_dataloader}")
    
    
    return train_dataloader,test_dataloader
    pass

# Generates SHAP values for all pixels in the MRIs given by the test_loader
def create_SHAP_values(bg_loader, test_loader, model, mri_count, save_path):
    '''
    Attributes:
        bg_loader (torch.utils.data.DataLoader): Dataloader instance for the background dataset.
        test_loader (torch.utils.data.DataLoader): Dataloader instance for the test dataset.
        mri_count (int): The total number of explanations to generate.
        save_path (str): The path to save the generated SHAP values (as .npy files).
    '''
   
    logger.info(f"save-path : {save_path}")
    if not os.path.exists(save_path): 
        os.mkdir(save_path) 

    # test_mri_files, test_paths, test_labels = next(iter(test_loader))
    train_mri_files, train_paths, train_labels = next(iter(bg_loader))
    test_load = iter(test_loader)
    train = torch.unsqueeze(train_mri_files,1)
   
    e = shap.DeepExplainer(model, train)                                                                           
    
    df = pd.read_csv("../data/datasets/ADNI3/all_file.csv") 
    
    for i in df["filename"]: 
        data = next(test_load)
        mri_test, test_path, test_label = data
        print(f"test_path: {test_path}")
        test_dim_inc = torch.unsqueeze(mri_test,1)
        shap_value = e.shap_values(test_dim_inc)
        np.save(save_path+"/"+i, shap_value)
    
# Aggregates SHAP values per brain region and returns a dictionary that maps 
# each region to the average SHAP value of its pixels. 
def aggregate_SHAP_values_per_region(shap_values, seg_path, brain_regions):
    '''
    Attributes:
        shap_values (ndarray): The shap values for an MRI (.npy).
        seg_path (str): The path to the segmented MRI (.nii). 
        brain_regions (dict): The regions inside the segmented MRI image (see data_util.py)
    '''
    
    # YOUR CODE HERE
    load_img = nib.load(seg_path)
    nii_data = load_img.get_fdata()
    
    shap_values1 = np.load(shap_values)
    
    dim_shap_values = shap_values1[0].squeeze(0).squeeze(0)
    
    shap_per_region = dict()
    
    for x in range(len(nii_data)):
        for y in range(len(nii_data[x])):
            for z in range(len(nii_data[x][y])):
                
                val_per_region = dim_shap_values[x][y][z]
                this_region = nii_data[x][y][z]
                
                if (this_region !=0):
                    if brain_regions[this_region] not in shap_per_region.keys():
                        shap_per_region[brain_regions[this_region]] = [val_per_region]
                    else:
                        shap_per_region[brain_regions[this_region]].append(val_per_region)
                        
    return shap_per_region
                        
    pass

# Returns a list containing the top-10 most contributing brain regions to each predicted class (AD/NotAD).
def output_top_10_lst(reg_dict,csv_file):
    '''
    Attribute:
        reg_dict (dict) : Dictionary with aggregated values per region
        csv_file (str): The path to a CSV file that will store the top 10 contributing regions.
    '''
    # YOUR CODE HERE
    
    sorted_dict ={k: v for k, v in sorted(reg_dict.items(), key=lambda item: item[1], reverse=True)}
    
    print(sorted_dict)
    count=0
    
    with open(csv_file, 'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in sorted_dict.items():
            if count==10:
                break
            writer.writerow([key, value])
            count+=1
    pass

# Plots SHAP values on a 2D slice of the 3D MRI. 
def plot_shap_on_mri(subject_mri, shap_values, filename):
    '''
    Attributes:
        subject_mri (str): The path to the MRI (.npy).
        shap_values (str): The path to the SHAP explanation that corresponds to the MRI (.npy).
    '''
    # YOUR CODE HERE
    
    mri_entry = np.load(subject_mri)

    shap_entry = np.load(shap_values)

    fig, (spatial, axial, coronal) = plt.subplots(1,3)
    orig_spatial = mri_entry[90, :, :]
    orig_axial = mri_entry[:, 108, :]
    orig_coronal = mri_entry[:, :, 90]

    spatial_shap_pos = shap_entry[1, 0, 0, :, 90, :]
    axial_shap_pos = shap_entry[1, 0, 0, :, 108, :]
    coronal_shap_pos = shap_entry[1, 0, 0, :, :, 90]

    spatial_shap_neg = shap_entry[0, 0, 0, :, 90, :]
    axial_shap_neg = shap_entry[0, 0, 0, :, 108, :]
    coronal_shap_neg = shap_entry[0, 0, 0, :, :, 90]

    plt.Figure(figsize=(100, 100))
    plt.style.use('grayscale')
    spatial = fig.add_subplot(1, 3, 1)

    plt.imshow(np.rot90(orig_spatial), cmap = 'gray')
    plt.imshow(np.rot90(spatial_shap_pos), cmap = red_transparent_blue)
    plt.imshow(np.rot90(spatial_shap_neg), cmap = red_transparent_blue)


    plt.Figure(figsize=(100, 100))
    plt.style.use('grayscale')
    axial = fig.add_subplot(1, 3, 2)

    plt.imshow(np.rot90(orig_axial), cmap = 'gray')
    plt.imshow(np.rot90(axial_shap_pos), cmap = red_transparent_blue)
    plt.imshow(np.rot90(axial_shap_neg), cmap = red_transparent_blue)

    plt.Figure(figsize=(100, 100))
    plt.style.use('grayscale')
    coronal = fig.add_subplot(1, 3, 3)

    plt.imshow(np.rot90(orig_coronal), cmap = 'gray')
    plt.imshow(np.rot90(coronal_shap_pos), cmap = red_transparent_blue)
    plt.imshow(np.rot90(coronal_shap_neg), cmap = red_transparent_blue)
    
    plt.show()
    
    try: 
        os.mkdir(output_folder+"/output/SHAP/heatmaps/") 
    except OSError as error: 
        print(error)
        
    plt.savefig(output_folder+"/output/SHAP/heatmaps/"+filename)
   
    pass


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--task',type=int, help='a task for the accumulator')
    parser.add_argument('--dataFolder', help='data folder filepath for the accumulator')
    parser.add_argument('--outputFolder', help='output folder filepath for the accumulator')
    
    args = parser.parse_args()
    task = args.task
    data_folder = args.dataFolder #../data/datasets/ADNI3
    output_folder = args.outputFolder #../data
    
    # TASK I: Load CNN model and isntances (MRIs)
    #         Report how many of the 19 MRIs are classified correctly
    # YOUR CODE HERE 
    if task == 1:
        model = _CNN(20,0.6)
        
        save_path = data_folder+'/cnn_best.pth'
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["state_dict"])
    
        lists = read_csv(data_folder+'/ADNI3.csv')
        df_bg, df_test=split_csv(lists[0],lists[1])
        
        train_dataloader,test_dataloader=prepare_dataloaders(bg_csv=data_folder+"/bg_file.csv", test_csv=data_folder+"/all_file.csv")
    
        test_mri_files, test_paths, test_labels = next(iter(test_dataloader))
        train_mri_files, train_paths, train_labels = next(iter(train_dataloader))
        
        logger.info(f"len1 : {test_mri_files.size()}")
        logger.info(f"len1-tr : {train_mri_files.size()}")
        
        test_correct = 0
        test_incorrect = 0
        test_total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            model.eval()
            for data in test_dataloader:
                
                test_mri_file, test_path, test_label = data
                
                output = model(torch.unsqueeze(test_mri_file,1))
            
                _, predicted = torch.max(output.data, 1)
                test_total += test_label
                for l in range(len(test_label)):
                    test_correct += (predicted == test_label[l]).item()
                    test_incorrect += (predicted != test_label[l]).item()
                    if predicted!=test_label[l]:
                        print(f"Incorrectly classified mri : {test_path[l]}")
        
        logger.info(f"Instances of test set classified correctly:{test_correct}")
        
        task1_df = {'Classified':["Correct","Incorrect"], 'Value':[test_correct, test_incorrect]}
        df = pd.DataFrame(task1_df)
        
        try: 
            os.mkdir(output_folder+"/output") 
        except OSError as error: 
            print(error)  
            
        df.to_csv(output_folder+"/output/task-1.csv",index=False)


    # TASK II: Probe the CNN model to generate predictions and compute the SHAP 
    #          values for each MRI using the DeepExplainer or the GradientExplainer. 
    #          Save the generated SHAP values that correspond to instances with a
    #          correct prediction into output/SHAP/data/
    # YOUR CODE HERE 
    if task ==2:
        mri_count=19
        try: 
            os.mkdir(output_folder+"/output/SHAP") 
        except OSError as error: 
            print(error)
            
        save_path = output_folder+"/output/SHAP/data/"
        
        create_SHAP_values(train_dataloader, test_dataloader, model, mri_count, save_path)

    # TASK III: Plot an explanation (pixel-based SHAP heatmaps) for a random MRI. 
    #           Save heatmaps into output/SHAP/heatmaps/
    # YOUR CODE HERE 
    if task == 3:
    
        save_path = output_folder+"/output/SHAP/data/"
        AD0_file = "ADNI_130_S_6612_MR_Sagittal_3D_Accelerated_MPRAGE_br_raw_20200205172217741_183_S920078_I1285682.npy"
        AD1_file = "ADNI_135_S_6545_MR_Accelerated_Sag_IR-FSPGR___br_raw_20190913133638192_107_S873086_I1226543.npy"
        AD0_shap = save_path + AD0_file
        AD1_shap = save_path + AD1_file
        
        AD0_mri = data_folder +"/"+ AD0_file
        AD1_mri = data_folder +"/"+ AD1_file
        heatmap_path = output_folder + "/output/SHAP/heatmaps"
        
        plot_shap_on_mri(AD0_mri, AD0_shap,"AD1-heatmap.png")
        plot_shap_on_mri(AD1_mri, AD1_shap, "AD0-heatmap.png")
    
    # # TASK IV: Map each SHAP value to its brain region and aggregate SHAP values per region.
    # #          Report the top-10 most contributing regions per class (AD/NC) as top10_{class}.csv
    # #          Save CSV files into output/top10/
    # # YOUR CODE HERE 
    
    if task == 4:
        # Incorrectly classified mri : ADNI_135_S_6389_MR_Accelerated_Sag_IR-FSPGR___br_raw_20190710091131883_145_S839966_I1185266.npy
        shap_dir_path = output_folder+"/output/SHAP/data/"
        df = pd.read_csv("../data/datasets/ADNI3/all_file.csv")
        filenames = list(df["filename"]) 
        regions_list_AD =[]
        regions_list_NOT_AD = []
        
        AD_dict = dict()
        NOT_AD_dict = dict()
        segpath = "../data/datasets/ADNI3/seg/"
        for AD_value in range(len(df["AD"])):
            
            if(df["AD"][AD_value] == 1 and filenames[AD_value] != "ADNI_135_S_6389_MR_Accelerated_Sag_IR-FSPGR___br_raw_20190710091131883_145_S839966_I1185266.npy"):
                shap_values_path = shap_dir_path + filenames[AD_value]
                seg_path = segpath + filenames[AD_value][:-3] + "nii"
                
                regions_aggregate = aggregate_SHAP_values_per_region(brain_regions=brain_regions, shap_values=shap_values_path, seg_path=seg_path)
                regions_list_AD.append(regions_aggregate)
                
            elif(df["AD"][AD_value] == 0 and filenames[AD_value] != "ADNI_135_S_6389_MR_Accelerated_Sag_IR-FSPGR___br_raw_20190710091131883_145_S839966_I1185266.npy"):
                
                shap_values_path = shap_dir_path + filenames[AD_value]
                seg_path = segpath + filenames[AD_value][:-3] + "nii"
                
                aggregate_SHAP_values_per_region(brain_regions=brain_regions, shap_values=shap_values_path, seg_path=seg_path)
                regions_aggregate = aggregate_SHAP_values_per_region(brain_regions=brain_regions, shap_values=shap_values_path, seg_path=seg_path)
                regions_list_NOT_AD.append(regions_aggregate)
                
        # print(f"regions_list_AD: {regions_list_AD}")        
        for region in range(1,96):
            for entry in regions_list_AD:
                if(brain_regions[region] not in AD_dict.keys()):
                    AD_dict[brain_regions[region]] = entry[brain_regions[region]]
                else:
                    AD_dict[brain_regions[region]].extend(entry[brain_regions[region]])
            
            for entry in regions_list_NOT_AD:
                if(brain_regions[region] not in NOT_AD_dict.keys()):
                    NOT_AD_dict[brain_regions[region]] = entry[brain_regions[region]]
                else:
                    NOT_AD_dict[brain_regions[region]].extend(entry[brain_regions[region]])  
                    
                    
        for reg in AD_dict:
            AD_dict[reg] = sum(AD_dict[reg])/len(AD_dict)
            
        for reg in NOT_AD_dict:
            NOT_AD_dict[reg] = sum(NOT_AD_dict[reg])/len(NOT_AD_dict)      
                
        print("AD_DICT :", AD_dict)
        print("\nNOT_AD_DICT :", NOT_AD_dict)
        
        top10_path = output_folder+"/output/"
        AD_top10 = output_top_10_lst(AD_dict, csv_file=top10_path+"task-4-true.csv")
        NOT_AD_top10 = output_top_10_lst(NOT_AD_dict, csv_file=top10_path+"task-4-false.csv")
        
        
        
    pass


