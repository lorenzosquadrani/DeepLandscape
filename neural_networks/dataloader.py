import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd

class pima_dataset(Dataset):

    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input = torch.tensor(self.data.iloc[idx, :8], dtype = torch.float)
        label = int(self.data.iloc[idx,8])
        return input, label
      
      
def get_dataloaders(batch_size, PATH):
  
  #GET DATA FROM FILE
  data = pd.read_csv(PATH + "pima-indians-diabetes.csv",
                     header = None, names =["Pregnancies","Glucose","BloodPressure","SkinThickness",
                                            "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"])
  
  traindata = data.sample(frac = 0.8 , random_state=1)
  testdata =data.drop(traindata.index)
  
  #BALANCING CATEGORIES (random re-sampling of the category with fewer samples)
  class0 = traindata[traindata["Outcome"]==0]
  class1 = traindata[traindata["Outcome"]==1]
  class1_over = class1.sample(len(class0), replace = True, random_state = 2)
  balanced_traindata = pd.concat([class0, class1_over])
  balanced_traindata = balanced_traindata.sample(frac = 1).reset_index(drop=True)
  
  #CREATE PYTORCH DATASETS
  trainset = pima_dataset(balanced_traindata)
  testset = pima_dataset(testdata)
  
  trainloader = DataLoader(trainset, batch_size =batch_size, shuffle = True)
  testloader = DataLoader(testset, batch_size = batch_size, shuffle = False)
  
  return trainloader, testloader
