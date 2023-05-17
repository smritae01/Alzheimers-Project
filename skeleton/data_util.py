import csv
import numpy as np
import os
import pandas as pd
import random
import torch
from torch.utils.data import Dataset

# Regions inside a segmented brain MRI (ONLY FOR TASK IV)
brain_regions = {1.:'TL hippocampus R',
                2.:'TL hippocampus L',
                3.:'TL amygdala R',
                4.:'TL amygdala L',
                5.:'TL anterior temporal lobe medial part R',
                6.:'TL anterior temporal lobe medial part L',
                7.:'TL anterior temporal lobe lateral part R',
                8.:'TL anterior temporal lobe lateral part L',
                9.:'TL parahippocampal and ambient gyrus R',
                10.:'TL parahippocampal and ambient gyrus L',
                11.:'TL superior temporal gyrus middle part R',
                12.:'TL superior temporal gyrus middle part L',
                13.:'TL middle and inferior temporal gyrus R',
                14.:'TL middle and inferior temporal gyrus L',
                15.:'TL fusiform gyrus R',
                16.:'TL fusiform gyrus L',
                17.:'cerebellum R',
                18.:'cerebellum L',
                19.:'brainstem excluding substantia nigra',
                20.:'insula posterior long gyrus L',
                21.:'insula posterior long gyrus R',
                22.:'OL lateral remainder occipital lobe L',
                23.:'OL lateral remainder occipital lobe R',
                24.:'CG anterior cingulate gyrus L',
                25.:'CG anterior cingulate gyrus R',
                26.:'CG posterior cingulate gyrus L',
                27.:'CG posterior cingulate gyrus R',
                28.:'FL middle frontal gyrus L',
                29.:'FL middle frontal gyrus R',
                30.:'TL posterior temporal lobe L',
                31.:'TL posterior temporal lobe R',
                32.:'PL angular gyrus L',
                33.:'PL angular gyrus R',
                34.:'caudate nucleus L',
                35.:'caudate nucleus R',
                36.:'nucleus accumbens L',
                37.:'nucleus accumbens R',
                38.:'putamen L',
                39.:'putamen R',
                40.:'thalamus L',
                41.:'thalamus R',
                42.:'pallidum L',
                43.:'pallidum R',
                44.:'corpus callosum',
                45.:'Lateral ventricle excluding temporal horn R',
                46.:'Lateral ventricle excluding temporal horn L',
                47.:'Lateral ventricle temporal horn R',
                48.:'Lateral ventricle temporal horn L',
                49.:'Third ventricle',
                50.:'FL precentral gyrus L',
                51.:'FL precentral gyrus R',
                52.:'FL straight gyrus L',
                53.:'FL straight gyrus R',
                54.:'FL anterior orbital gyrus L',
                55.:'FL anterior orbital gyrus R',
                56.:'FL inferior frontal gyrus L',
                57.:'FL inferior frontal gyrus R',
                58.:'FL superior frontal gyrus L',
                59.:'FL superior frontal gyrus R',
                60.:'PL postcentral gyrus L',
                61.:'PL postcentral gyrus R',
                62.:'PL superior parietal gyrus L',
                63.:'PL superior parietal gyrus R',
                64.:'OL lingual gyrus L',
                65.:'OL lingual gyrus R',
                66.:'OL cuneus L',
                67.:'OL cuneus R',
                68.:'FL medial orbital gyrus L',
                69.:'FL medial orbital gyrus R',
                70.:'FL lateral orbital gyrus L',
                71.:'FL lateral orbital gyrus R',
                72.:'FL posterior orbital gyrus L',
                73.:'FL posterior orbital gyrus R',
                74.:'substantia nigra L',
                75.:'substantia nigra R',
                76.:'FL subgenual frontal cortex L',
                77.:'FL subgenual frontal cortex R',
                78.:'FL subcallosal area L',
                79.:'FL subcallosal area R',
                80.:'FL pre-subgenual frontal cortex L',
                81.:'FL pre-subgenual frontal cortex R',
                82.:'TL superior temporal gyrus anterior part L',
                83.:'TL superior temporal gyrus anterior part R',
                84.:'PL supramarginal gyrus L',
                85.:'PL supramarginal gyrus R',
                86.:'insula anterior short gyrus L',
                87.:'insula anterior short gyrus R',
                88.:'insula middle short gyrus L',
                89.:'insula middle short gyrus R',
                90.:'insula posterior short gyrus L',
                91.:'insula posterior short gyrus R',
                92.:'insula anterior inferior cortex L',
                93.:'insula anterior inferior cortex R',
                94.:'insula anterior long gyrus L',
                95.:'insula anterior long gyrus R',
}

# The dataset
class CNN_Data(Dataset):
    '''
        This is a custom dataset that inherits from torch.utils.data.Dataset. 
    '''
    def __init__(self, csv_dir):
        '''
        Attributes:
            csv_dir (str): The path to the CSV file that contains the MRI metadata.
        '''
        self.csv_dir = csv_dir
        self.data = pd.read_csv(csv_dir)
        
        # YOUR CODE HERE 
        pass

    # Returns total number of data samples
    def __len__(self):
        # YOUR CODE HERE 
        length = len(self.data)
        return length
        pass

    # Returns the actual MRI data, the MRI filename, and the label
    def __getitem__(self, idx):
        '''
        Attribute:
            idx (int): The sample MRI index.
        '''
        # YOUR CODE HERE 
        path = ".."+self.data.iloc[idx, 0]+self.data.iloc[idx, 1]
        p=self.data.iloc[idx, 1]
        # print("path =",path)
        mri_file = np.load(path)
        label = self.data.iloc[idx, 12]
        return mri_file, p, label
        
        pass

# This is a helper function that performs the following steps:
#   1. Retrieves the metadata for the 19 MRIs provided 
#   2. Splits the 19 MRIs into two randomly selected datasets: 
#    - One that will be used for probing/testing the model (make sure it contains at least 5 MRIs).
#    - One the will be used as a background dataset for SHAP
# The function creates two new CSV files containing the metadata for each of the above datasets.
def split_csv(paths, labels, output_folder='../data/datasets', random_seed = 1051):
    '''
    Attributes:
        csv_file (str): The path to the CSV file that contains the MRI metadata.
        output_folder (str): The path to store the CSV files for the test and background datasets.
        random_seed (int): The seed number to shuffle the csv_file (you can also define your own seed).
    '''
    # YOUR CODE HERE 
    mri_metadata = []
    for filename in os.listdir(output_folder+"/ADNI3"):
        if filename.endswith(".npy"):
            for i in range(len(paths)):
                if filename == paths[i]:
                    mri_metadata.append(labels[i])
                    
    header = labels[0]
    
    random.seed(random_seed)
    random.shuffle(mri_metadata)
    
    # bg_file.csv has 13 and test_file.csv has 6 entries
    
    df = pd.DataFrame(mri_metadata[6:], columns=header)
    
    df.to_csv(output_folder+"/ADNI3/bg_file.csv",index=False)
        
    df2 = pd.DataFrame(mri_metadata[0:6], columns=header)
    
    df2.to_csv(output_folder+"/ADNI3/test_file.csv",index=False)
    
    df_all = pd.DataFrame(mri_metadata[:], columns=header)
    
    df_all.to_csv(output_folder+"/ADNI3/all_file.csv",index=False)
       
    return df, df2
    pass

# Returns one list containing the MRI filepaths and a second list with the respective labels
def read_csv(filename):
    '''
    Attributes:
        filename (str): The path to the CSV file that contains the MRI metadata.
    '''
    # YOUR CODE HERE 
    paths_list = []
    labels_list = []
    
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    for i in range(len(data)):
        paths_list.append(data[i][1])
        labels_list.append(data[i])    

    return [paths_list, labels_list]
    
    pass
