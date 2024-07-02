import pandas as pd
import torch 
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
from torch import distributed
import os
import argparse

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
    

class SonarDataset(Dataset):
  def __init__(self, src_file, root_dir, transform=None):
    sonarDataset = pd.read_csv(src_file, header=None)

    X = sonarDataset.iloc[:, :60]
    Y = sonarDataset.iloc[:, 60]

    nomeClases = Y.unique()
    YConversion = pd.DataFrame()

    for nome in nomeClases:
      YConversion[nome] = (Y==nome).apply(lambda x : 1.0 if x else 0.0)
      
    y_tensor = torch.as_tensor(YConversion.to_numpy()).type(torch.float32)

    df_dict = dict.fromkeys(X.columns, '')
    X.rename(columns = df_dict)
    s1=X.iloc[:, :60].values
    x_tensor = torch.tensor(s1)

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

    preds = self.data[idx, 0:60]
    spcs = self.data[idx, 60:]
    sample = (preds, spcs)
    
    if self.transform:
      sample = self.transform(sample)
    return sample
  

class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(in_features=50, out_features=2)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x
    

def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    
    for i, data in enumerate(train_ldr):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        
        if i % 10 == 9:
            last_loss = running_loss / 10
            running_loss = 0.
            
    return last_loss


# Prueba carga de datos
dataset = SonarDataset("sonar.all-data",".")

# División en train y test
lonxitudeDataset = len(dataset)

tamTrain =int(lonxitudeDataset*0.8)
tamVal = lonxitudeDataset - tamTrain

train_set, val_set = random_split(dataset,[tamTrain,tamVal])

train_ldr = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True, drop_last=False)

validation_loader =torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False, num_workers=2)


rank = int(os.environ["SLURM_PROCID"])
world_size = int(os.environ["WORLD_SIZE"])


if world_size > 1:
    distributed.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
    )

# Instanciamos el modelo
model = Model(60)
if distributed_is_initialized():
    model = nn.parallel.DistributedDataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn   = nn.CrossEntropyLoss()


entradaProba,dest = next(iter(train_ldr))

print("Entrada:")
print(entradaProba)

print("Desexada:")
print(dest)

saida = model(entradaProba) 

print("Saída:")
print(saida)

loss_fn(saida, dest)


EPOCHS = 10
loss_list     = torch.zeros((EPOCHS,))
accuracy_list = torch.zeros((EPOCHS,))

metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2)

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    model.train(True)
    avg_loss = train_one_epoch(epoch, None)
    loss_list[epoch] = avg_loss
    
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        print(f"Entrada: {vinputs}")

        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)

        print(f"\nSalidas deseadas: {vlabels}")
        print(f"\nSalidas: {voutputs}")
        print("<---------------------------------------------------------->")

        acc = metric(voutputs, vlabels)

        running_vloss += vloss

    acc = metric.compute()
    print(f"Accuracy on epoch: {acc}\n")
    
