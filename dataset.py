import torch 
from torch.utils import data
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


# adjust the number of channels before running or evaluaing the model 

#BrainWaves_to_get_custom_seconds_data
class BrainWaves1(data.Dataset):
    def __init__(self, kind = "train", k = 1, balancing = "smote", normalize = False, channels = 8):

        self.labels = "./labels_all_data.csv"

        self.df = pd.read_csv(self.labels)

        self.df = self.df[:9444]
        file_paths = self.df["path"]
        labels = self.df["label"]
        
        # getting all the features in the same np file 
        r_features = np.array([np.load(file_path + ".npy", allow_pickle= 1) for file_path in file_paths])

        if channels == 3:
            r_features = r_features[:,2:5, 250:1000]
        elif channels == 2:
            r_features = r_features[:,:channels, 250:1000]
        else:
            r_features = r_features[:,:channels, 250:1000]
        # reshaping to match input requirements for smote 
        r_features = r_features.reshape(-1, channels * 750)

    
        smote = SMOTE(random_state=42)
        self.features, self.labels = smote.fit_resample(r_features, labels)
        
        self.features = self.features.reshape(-1, channels, 750)
        
        # np.save("s250-1000-smote_until_9444.npy", self.features)
        # self.labels.to_csv('labels_smote')

        indices = list(range(len(self.features)))
        np.random.seed(2)
        np.random.shuffle(indices)
        self.size = len(indices)


        if kind == 'train':
            self.indices = indices[:int((k - 1) * self.size / 10)] + indices[int(k * self.size / 10):]
        if kind == 'val':
            self.indices = indices[int((k - 1) * self.size / 10):int(k * self.size / 10)]
        if kind == 'all':
            self.indices = indices

        self.normalize = normalize
        # print(self.indices)
        print("main_job done")

    def __getitem__(self, indx):

        # data = torch.tensor().float(dtype = torch.float64)
        # np_array = np.load(f'{self.labels_df.iloc[self.indices[indx]]["path"]}.npy', allow_pickle= 1)
        # np_array = np.array(np_array.tolist(), dtype=np.float32)

        # Convert to PyTorch tensor
        data = torch.tensor(self.features[self.indices[indx]]).float()  # Use float32 for tensor
        
        label = self.labels[self.indices[indx]]

        if self.normalize == True: 
            mean = data.mean(dim=1, keepdim=True)
            std = data.std(dim=1, keepdim=True)
            # print(bio_data.sum(dim = 1, keepdim = True))
            # print(bio_data.std())
            # print(bio_data.sum())
            # Normalize each row (channel-wise)
            epsilon = 1e-8  # Small constant to avoid division by zero
            bio_data = (data - mean) / (std + epsilon)
        
            
            return bio_data, label
        
        return data, label
    
    def __len__(self):
        return len(self.indices)



class BrainWaves(data.Dataset):
    def __init__(self, kind = "train", k = 1, balancing = "smote"):
        self.labels = "./labels_smote"

        self.df = pd.read_csv(self.labels)

        
        self.labels = self.df["label"]
        
        self.features = np.load("s250-1000-smote.npy")
        
        

        indices = list(range(len(self.features)))
        # print(len(self.features))
        # indices = list(range(1000))
        # np.random.seed(3)
        np.random.shuffle(indices)
        self.size = len(indices)

        

        if kind == 'train':
            self.indices = indices[:int((k - 1) * self.size / 10)] + indices[int(k * self.size / 10):]
        if kind == 'val':
            self.indices = indices[int((k - 1) * self.size / 10):int(k * self.size / 10)]
        if kind == 'all':
            self.indices = indices

        print("main_job done")
        # print(self.indices)

    def __getitem__(self, indx):

        # data = torch.tensor().float(dtype = torch.float64)
        # np_array = np.load(f'{self.labels_df.iloc[self.indices[indx]]["path"]}.npy', allow_pickle= 1)
        # np_array = np.array(np_array.tolist(), dtype=np.float32)

        # Convert to PyTorch tensor
        data = torch.tensor(self.features[self.indices[indx]]).float()  # Use float32 for tensor
        
        label = self.labels[self.indices[indx]]

        return data, label
    
    def __len__(self):
        return len(self.indices)



#BrainWaves_Test dataset
class BrainWavesTest(data.Dataset):
    def __init__(self, kind = "train", k = 1, balancing = "smote", normalize = 0):
        self.labels = "./independent_subject_test.csv"

        self.df = pd.read_csv(self.labels)

        file_paths = self.df["path"]
        self.labels = self.df["label"]
        
        # getting all the features in the same np file 
        r_features = np.array([np.load(file_path + ".npy", allow_pickle= 1) for file_path in file_paths])

        r_features = r_features[:,:, 250:1000]
        # reshaping to match input requirements for smote 
        r_features = r_features.reshape(-1, 8 * 750)

        
        self.features = r_features.reshape(-1, 8, 750)
        
        
        indices = list(range(len(self.features)))
        # np.random.seed(2)
        self.size = len(indices)


        self.indices = indices

    def __getitem__(self, indx):
        data = self.features[self.indices[indx]]

        data = np.array(data, dtype=np.float32)

        # Create a PyTorch tensor
        data = torch.tensor(data, dtype=torch.float32) # Use float32 for tensor
        # print(data)
        label = self.labels[self.indices[indx]]
        
        if self.normalize == True: 
            mean = data.mean(dim=1, keepdim=True)
            std = data.std(dim=1, keepdim=True)
            # print(bio_data.sum(dim = 1, keepdim = True))
            # print(bio_data.std())
            # print(bio_data.sum())
            # Normalize each row (channel-wise)
            epsilon = 1e-8  # Small constant to avoid division by zero
            bio_data = (data - mean) / (std + epsilon)
        
            
            return bio_data, label
        return data, label
    
    def __len__(self):
        return len(self.indices)

#BrainWaves_to_get_custom_seconds_data
class BrainWavesTest_v2(data.Dataset):
    def __init__(self, kind = "all", k = 1, normalize = False, channels = 8):

        self.labels = "./labels_all_data.csv"

        self.df = pd.read_csv(self.labels)

        self.df = self.df[9445:]
        file_paths = self.df["path"]
        self.labels = list(self.df["label"])

        # print(self.labels)
        # print(len(self.df), len(file_paths), len(self.labels))
        # print(self.labels)
        # getting all the features in the same np file 
        r_features = np.array([np.load(file_path + ".npy", allow_pickle= 1) for file_path in file_paths])


        r_features = r_features[:,:, 250:1000]
    

        self.features = r_features.reshape(-1, 8, 750)

        if channels == 2:
            self.features = self.features[:, :2, :]
        self.indices = list(range(len(self.features)))
        # np.random.seed(2)
        
        self.size = len(self.indices)
        self.normalize = normalize


        
        print("main_job done")

    def __getitem__(self, indx):

        # data = torch.tensor().float(dtype = torch.float64)
        # np_array = np.load(f'{self.labels_df.iloc[self.indices[indx]]["path"]}.npy', allow_pickle= 1)
        # np_array = np.array(np_array.tolist(), dtype=np.float32)

        
        data = self.features[self.indices[indx]]

        data = np.array(data, dtype=np.float32)

        # Create a PyTorch tensor
        data = torch.tensor(data, dtype=torch.float32) # Use float32 for tensor
        
        
        label = self.labels[self.indices[indx]]
        

        if self.normalize == True: 
            mean = data.mean(dim=1, keepdim=True)
            std = data.std(dim=1, keepdim=True)
            # print(bio_data.sum(dim = 1, keepdim = True))
            # print(bio_data.std())
            # print(bio_data.sum())
            # Normalize each row (channel-wise)
            epsilon = 1e-8  # Small constant to avoid division by zero
            bio_data = (data - mean) / (std + epsilon)
            # print(bio_data.shape)
            
            return bio_data, label, self.df.iloc[indx]["path"]
        return data, label, self.df.iloc[indx]["path"]
    
    def __len__(self):
        return self.size