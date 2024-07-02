import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torchmetrics import MeanSquaredError, MeanAbsoluteError
import lightning.pytorch as pl

import warnings
warnings.filterwarnings("ignore")


class StandardScaler:

    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def __repr__(self):
        return f"mean: {self.mean}, std:{self.std}, epsilon:{self.epsilon}"


class AutomobileDataset(Dataset):
  def __init__(self, src_file, root_dir, transform=None):
    automobileDataset = pd.read_csv(src_file, header=None)

    automobileDataset.replace('?', pd.NA, inplace=True)
    automobileDataset.dropna(subset=automobileDataset.columns, inplace=True)

    columnas_categoricas = [2, 3, 4, 5, 6, 7, 8, 14, 15, 17]

    class_label_encoder = LabelEncoder()

    for i in columnas_categoricas:
        automobileDataset[i] = class_label_encoder.fit_transform(automobileDataset[i])

    automobileDataset[[1, 18, 19, 21, 22, 25]] = automobileDataset[[1, 18, 19, 21, 22, 25]].astype(float)

    X = automobileDataset.iloc[:, :25]
    Y = automobileDataset.iloc[:, 25]
    
    x1=X.iloc[:,0:25].values
  
    x_tensor = torch.tensor(x1)
    y_tensor = torch.tensor(Y.values).type(torch.float32).unsqueeze(1)

    scaler = StandardScaler()
    
    XScalada = scaler.fit_transform(x_tensor).type(torch.float32)

    self.data = torch.cat((XScalada,y_tensor),1)
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    preds = self.data[idx, 0:25]
    spcs = self.data[idx, 25]
    sample = (preds, spcs)
    
    if self.transform:
      sample = self.transform(sample)
    return sample

class Model(pl.LightningModule):
  def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 100)
        self.layer2 = nn.Linear(100, 50)
        self.layer3 = nn.Linear(50, 1)
        
  def forward(self, x):
      x = F.relu(self.layer1(x))
      x = F.relu(self.layer2(x))
      x = self.layer3(x)
      return x

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer 
  
  def training_step(self, train_batch, batch_idx):
    inputs, labels = train_batch
    
    outputs = model(inputs)
    
    loss = loss_fn(outputs, labels)
    self.log('"Train/Loss"', loss)

    return loss
  
  def validation_step(self, val_batch, batch_idx):
    vinputs, vlabels = val_batch
    voutputs = model(vinputs).flatten()

    mean_squared_error = MeanSquaredError()
    mean_absolute_error = MeanAbsoluteError()

    mean_squared_error(voutputs,vlabels)
    mean_absolute_error(voutputs,vlabels)

    loss = loss_fn(voutputs, vlabels)

    errorMedio = mean_squared_error.compute()
    errorAbsolute =mean_absolute_error.compute()

    self.log('Test/Loss', loss)
    self.log("Test/Mean squared error", errorMedio)
    self.log("Test/Mean absolute error", errorAbsolute)



# Prueba carga de datos 
automobileDataset = AutomobileDataset("imports-85.data",".")
print(automobileDataset[0])

# División en train y test
lonxitudeDataset = len(automobileDataset)

tamTrain =int(lonxitudeDataset*0.8)
tamVal = lonxitudeDataset - tamTrain

train_set, val_set = random_split(automobileDataset,[tamTrain,tamVal])
train_ldr = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True, drop_last=False)
validation_loader =torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False, drop_last=True)

# Creación del modelo
model = Model(25)
loss_fn   = nn.MSELoss(reduction='sum')

# Entrenamiento del modelo
trainer = pl.Trainer(accelerator="cpu",num_nodes=2, devices=2, precision=32, 
                     limit_train_batches=0.5, max_epochs=100)
trainer.fit(model, train_ldr, validation_loader)


























