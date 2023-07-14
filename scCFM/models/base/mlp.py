import torch

class MLP(torch.nn.Module):
    def __init__(self, 
                 hidden_dims: list,
                 batch_norm: bool, 
                 dropout: bool, 
                 dropout_p: float, 
                 activation=torch.nn.ReLU):
        
        super(MLP, self).__init__()

        # Attributes 
        self.hidden_dims = hidden_dims
        self.batch_norm = batch_norm
        self.activation = activation

        # MLP 
        layers = []
        for i in range(len(self.hidden_dims[:-1])):
            block = []
            block.append(torch.nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            if batch_norm: 
                block.append(torch.nn.BatchNorm1d(self.hidden_dims[i+1]))
            block.append(self.activation())
            if dropout:
                block.append(torch.nn.Dropout(dropout_p))
            layers.append(torch.nn.Sequential(*block))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
if __name__=="__main__":
    mlp = MLP([2000, 256, 256], 
              batch_norm=True,
              dropout=True,
              dropout_p=0.2, 
              activation=torch.nn.ReLU)
    print(mlp)
    
        