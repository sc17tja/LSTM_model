import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import math
from sklearn.metrics import mean_squared_error
pd.options.mode.chained_assignment = None


def load_dataset(lookback):
    
    #Data pre-processing
    
    dataset = pd.read_csv(r"dataset\GBPUSD1min.csv", header=0) #Load dataset into a pandas DataFrame
    
    dataset.Datetime = pd.to_datetime(dataset.Datetime, format = '%Y-%m-%d/%H:%M:%S')
    
    dataset.set_index(dataset['Datetime'], inplace=True) #Indexes by the timestamp
        
    #Split into test and training datasets
    timeDiff = dataset.index[-1] - dataset.index[0]
    days = int(timeDiff.days)  
    splitday = dataset.index[0] + timedelta(days=days/2)
    train = dataset.loc[dataset.index < splitday]
    test = dataset.loc[dataset.index >= splitday] 

    #Normalise the datasets
    scaler = MinMaxScaler(feature_range=(0, 1))
        
    train[['Close']] = scaler.fit_transform(train[['Close']])
    test[['Close']] = scaler.transform(test[['Close']])
        
    train_seq = []
    train_target = []
    train_data = train.Close #Drop all data but the close prices
    
    #Constructs the training sequences with the desired lookback
    for i in range(len(train)-lookback): 
       if train_data.index[i+lookback] - train_data.index[i] == timedelta(minutes = lookback):
            train_seq.append(train_data.values[i:i+lookback])
            train_target.append(train_data.values[i+lookback])
    
    test_seq = []
    test_target = []
    test_data = test.Close #Drop all data but the close prices
   
    #Constructs the testing sequences with the desired lookback
    for i in range(len(test)-lookback):
        if test_data.index[i+lookback] - test_data.index[i] == timedelta(minutes = lookback):
            test_seq.append(test_data.values[i:i+lookback])
            test_target.append(test_data.values[i+lookback])
        
    #Convert to numpy array
    train_seq, train_target = np.array(train_seq), np.array(train_target)
    test_seq, test_target = np.array(test_seq), np.array(test_target)
    
    #Reshape so the model can read the sequences
    train_seq = np.reshape(train_seq, (train_seq.shape[0],train_seq.shape[1], 1))
    test_seq = np.reshape(test_seq, (test_seq.shape[0], test_seq.shape[1], 1))
    train_target = np.reshape(train_target, (train_seq.shape[0], 1))
    test_target = np.reshape(test_target, (test_seq.shape[0], 1))
    
    #Convert to tensors so the model can operate on the data
    train_seq = torch.from_numpy(train_seq).type(torch.Tensor)
    test_seq = torch.from_numpy(test_seq).type(torch.Tensor)
    train_target = torch.from_numpy(train_target).type(torch.Tensor)
    test_target = torch.from_numpy(test_target).type(torch.Tensor)
       
        
    return train_seq, train_target, test_seq, test_target, scaler


class LSTM(nn.Module):
    #Model definition
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__() #Inherits from Pytorches LSTM class

        self.hidden_dim = hidden_dim #Set hidden dimensions

        self.num_layers = num_layers #Set number of layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True) #Definie structure

        self.fc = nn.Linear(hidden_dim, output_dim) #Define output node

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Detach so it doesn't backpropagate all the way to the start after a batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        out = self.fc(out[:, -1, :]) 
        
        return out
    
#Hyperparameters
lookback = 15
hidden_layer_size = 64
num_layers = 3
batch_size = 16
num_epochs = 130
learning_rate = 0.001

#Define shape of input and output to 1 as there is only one input varibale and one varibale being predicted
input_size = 1
output_size = 1

#Fetch, pre-process and load the datasets into dataloaders
train_seq, train_target, test_seq, test_target, scaler = load_dataset(lookback) 

train = torch.utils.data.TensorDataset(train_seq, train_target)
test = torch.utils.data.TensorDataset(test_seq, test_target)

train_loader = torch.utils.data.DataLoader(dataset=train, 
                                           batch_size=batch_size, 
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test, 
                                          batch_size=batch_size, 
                                          shuffle=False)

model = LSTM(input_size, hidden_layer_size, num_layers, output_size) #Create instance of model
loss_fn = nn.MSELoss() #Define loss function as Mean Squared Error
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate) #Define optimiser as Adam optimiser

for t in range(num_epochs):
    
    # Forward pass
    y_train_pred = model(train_seq)

    loss = loss_fn(y_train_pred, train_target)
    if t % 10 == 0 and t !=0:
        print("Epoch ", t, "Loss (MSE): ", loss.item()) #Print loss every 10 epochs

    # Zero out gradient, else it will carry over epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update weights
    optimiser.step()


y_test_pred = model(test_seq) #Test the trained model

# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
train_target = scaler.inverse_transform(train_target.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
test_target = scaler.inverse_transform(test_target.detach().numpy())

# calculate root mean squared errors
trainScore = math.sqrt(mean_squared_error(train_target[:,0], y_train_pred[:,0]))
print('Train Score: %.5f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_target[:,0], y_test_pred[:,0]))
print('Test Score: %.5f RMSE' % (testScore))    