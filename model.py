import torch.nn as nn
import torch
import time
class Transformer1d(nn.Module):
    def __init__(self, input_size, n_classes, n_length, d_model, nhead, dim_feedforward, dropout, activation='relu'):
        super(Transformer1d, self).__init__()

        self.input_size = input_size
        self.n_classes = n_classes
        self.n_length = n_length
        self.d_model = d_model

        # Assuming input_size is not necessarily equal to d_model
        self.input_projection = nn.Linear(input_size, d_model)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Calculate the output size after the Transformer
        # This step is crucial and needs to be adjusted based on how you handle the sequence
        # For simplicity, assuming the output size remains d_model * n_length
        # This might need adjustment based on your actual sequence handling (e.g., pooling, averaging)
        self.fc_out_size = d_model * n_length
        # print("fc_out", self.fc_out_size)
        # Final linear layer to match the desired feature size (n_classes)
        self.fc = nn.Linear(self.fc_out_size, n_classes)

    def forward(self, x):
        # Input x shape: (batch_size, input_size, n_length)

        # Project input to d_model dimension
        x = x.permute(2, 0, 1)  # Change shape to (n_length, batch_size, input_size)
        x = self.input_projection(x)  # Shape becomes (n_length, batch_size, d_model)

        # Pass through the Transformer encoder
        x = self.transformer_encoder(x)  # Shape remains (n_length, batch_size, d_model)

        # Flatten the output
        x = x.permute(1, 2, 0)  # Change shape to (batch_size, d_model, n_length)
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch_size, d_model * n_length)
        # print(x.shape, "oyasid hpd asodih")
        # Final linear layer
        x = self.fc(x)  # Shape becomes (ba||tch_size, n_classes)

        return x


class Bio(nn.Module):
    def __init__(self, input_size=32, feature_size=64):
        super(Bio, self).__init__()

        self.features = Transformer1d(
            input_size= 8, # for brainwaves data
            n_classes=64,
            n_length=750,
            d_model=32,
            nhead=8,
            dim_feedforward=128,
            dropout=0.3,
            activation='relu'
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=1),
            nn.Linear(64, 20),
            nn.ReLU(inplace= 1),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        # x = self.cnn1d(x)
        x = self.features(x)
        x = self.classifier(x)
        x = x.squeeze(-1)
        return x
    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'physiological_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))



class Conv1D_v1(nn.Module):
    def __init__(self, channels = 8):
        super(Conv1D_v1, self).__init__() 

        self.seq = nn.Sequential(
         
         nn.Conv1d(channels, 16, kernel_size= 3),
         nn.ReLU(),
         nn.MaxPool1d(kernel_size= 2), 

        nn.Conv1d(16, 32, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size= 2),

        nn.Conv1d(32, 64, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size= 2),

        nn.Conv1d(64, 128, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size = 2)
        )

        if channels == 8:
            out_features = 5760
        if channels == 2: 
            out_features = 1
        self.classifier = nn.Sequential(
            nn.Linear(out_features, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 1000),
            nn.ReLU(inplace=1),
            nn.Linear(1000, 80),
            nn.ReLU(inplace= 1),
            nn.Linear(80, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
      
        x = self.seq(x)
      
        x = x.view(x.size(0), -1)
       
        x = self.classifier(x)

        return x
    
    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'physiological_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))

class Conv1D_v2(nn.Module):
    def __init__(self, channels = 8):
        super(Conv1D_v2, self).__init__() 

        self.seq = nn.Sequential(
         
         nn.Conv1d(channels, 16, kernel_size= 3),
         nn.ReLU(),
         nn.MaxPool1d(kernel_size= 2), 

        nn.Conv1d(16, 32, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size= 2),

        nn.Conv1d(32, 64, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size= 2),

        nn.Conv1d(64, 128, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size = 2),

        nn.Conv1d(128, 256, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size = 2)


        )

        out_features = 5376
        self.classifier = nn.Sequential(
            nn.Linear(out_features, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 1000),
            nn.ReLU(inplace=1),
            nn.Linear(1000, 80),
            nn.ReLU(inplace= 1),
            nn.Linear(80, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
      
        x = self.seq(x)
      
        x = x.view(x.size(0), -1)
       
        x = self.classifier(x)

        return x
    
    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'physiological_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))



class Conv1D_v3(nn.Module):
    def __init__(self):
        super(Conv1D_v2, self).__init__() 

        self.seq = nn.Sequential(
         
         nn.Conv1d(8, 16, kernel_size= 3),
         nn.ReLU(),
         nn.MaxPool1d(kernel_size= 2), 

        nn.Conv1d(16, 32, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size= 2),

        nn.Conv1d(32, 64, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size= 2),

        nn.Conv1d(64, 128, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size = 2),

        nn.Conv1d(128, 256, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size = 2)


        )

        
        self.classifier = nn.Sequential(
            nn.Linear(5376, 2000),
            nn.Dropout(p = 0.3), 
            nn.ReLU(inplace=True),
            nn.Linear(2000, 1000),
            nn.Dropout(p = 0.3), 
            nn.ReLU(inplace=1),
            nn.Linear(1000, 80),
            nn.Dropout(p = 0.3), 
            nn.ReLU(inplace= 1),
            nn.Linear(80, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
      
        x = self.seq(x)
      
        x = x.view(x.size(0), -1)
       
        x = self.classifier(x)

        return x
    
    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'physiological_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))






class Conv1D_v4(nn.Module):
    def __init__(self, channels, kernel_size = 3, padding = 1, stride = 2):
        super(Conv1D_v4, self).__init__() 

        self.seq = nn.Sequential(
         
            self.get_layer(channels, channels * 2, stride = stride, padding = padding, final_layer= False, kernel_size= kernel_size ),
            
            self.get_layer(channels * 2, channels * 4, stride = stride, padding = padding, final_layer= False, kernel_size= kernel_size),
            
            self.get_layer(channels * 4, channels * 2, stride = stride, padding = padding, final_layer= False, kernel_size= kernel_size),
            
            self.get_layer(channels * 2, channels, stride = stride, padding = padding, final_layer= False, kernel_size= kernel_size),
            
            self.get_layer(channels , 1, stride = 1, padding = 1, final_layer= 1, kernel_size= 2),

            # self.get_layer(channels//2, 1, stride = stride, padding = padding, final_layer= True, kernel_size= kernel_size),

        )

        


    def get_layer(self, input_c, output_c, stride, padding, final_layer, kernel_size):

        if not final_layer:
            return nn.Sequential(
                nn.Conv1d(input_c, output_c, kernel_size=kernel_size,padding = padding, stride = stride),
                nn.BatchNorm1d(output_c),
                nn.MaxPool1d(stride = stride, kernel_size=kernel_size),
                nn.LeakyReLU(0.2)
                
            )

        return nn.Sequential(
            nn.Conv1d(input_c, output_c, kernel_size=kernel_size,padding = padding, stride = stride),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
      
        x = self.seq(x)
      
        # print(x.shape)
        return x
    
    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'physiological_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))

