import torch
import torch.nn as nn
import numpy as np


class MLPmodel(nn.Module):
    # classification (transit/non-transit) for entire input segment
    def __init__(self, n_inputs, hidden_units, neg_slope=0, one_out=True):
        super(MLPmodel, self).__init__()
        hidden_units = hidden_units if isinstance(hidden_units, list) else [hidden_units]
        nodes = [n_inputs] + hidden_units + [1] if one_out else [n_inputs] + hidden_units
        self.linears = nn.ModuleList([nn.Linear(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)])
        self.LReLU = nn.LeakyReLU(neg_slope)

    def forward(self, x):
        # input x: [B, T]
        for layer in self.linears[:-1]:
            x = self.LReLU(layer(x))
        out = self.linears[-1](x).squeeze(dim=-1)
        return torch.sigmoid(out), out # [B,]

    
class CNNmodel(nn.Module):
    # classification (transit/non-transit) for entire input segment
    def __init__(self, n_inputs, channels, kernels, strides, fc_hiddens, pool=3, neg_slope=0, 
                 batch_norm=True, info=True):
        super(CNNmodel, self).__init__()
        channels = [1]+channels
        if channels[-1] != 1:
            print("WARNING: use channel of 1 for last conv layer")
        if batch_norm:
            self.modlist = nn.ModuleList([nn.Sequential(nn.Conv1d(c, channels[i+1], kernels[i], strides[i]),
                                 nn.BatchNorm1d(channels[i+1]), nn.LeakyReLU(neg_slope),
                                 nn.MaxPool1d(pool, stride=None)) for i,c in enumerate(channels[:-1])])
        else:
           # observation: without batch_norm, the network occasionally gets stuck at poor minima
            self.modlist = nn.ModuleList([nn.Sequential(nn.Conv1d(c, channels[i+1], kernels[i], strides[i]),
                                nn.LeakyReLU(neg_slope), nn.MaxPool1d(pool, stride=None)) for i,c in enumerate(channels[:-1])])
        for i in range(len(kernels)):
            n_inputs = int((n_inputs-(kernels[i]-1))/strides[i]/pool)
        print(f"{n_inputs} inputs left after conv layers") if info else "" 
        self.fc_inputs = n_inputs
        self.fc = MLPmodel(self.fc_inputs, fc_hiddens, neg_slope)
         
    def forward(self, x):
        # input x: [B, T]
        x = x.view(x.size(0), 1, -1)
        for mod in self.modlist:
            x = mod(x)
        out, logits = self.fc(x.view(-1,self.fc_inputs)) # sigmoid included
        return out, logits  # [B,]


class NaiveRNNmodel(nn.Module):
    # classification (transit/non-transit) for entire input segment
    def __init__(self, rnn_hiddens, fc_hiddens, neg_slope=0, lstm=False, bidirectional=True, num_layers=1):
        super(NaiveRNNmodel, self).__init__()
        if lstm:
            self.rnn = nn.LSTM(input_size=1, hidden_size=rnn_hiddens, num_layers=num_layers,
                               batch_first=True, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(input_size=1, hidden_size=rnn_hiddens, num_layers=num_layers,
                              batch_first=True, bidirectional=bidirectional)
        self.bidirectional = bidirectional
        self.rnn_hiddens = rnn_hiddens
        self.fc_inputs = rnn_hiddens*2 if bidirectional else rnn_hiddens
        self.fc = MLPmodel(self.fc_inputs, fc_hiddens, neg_slope)

    def forward(self, x):
        # input x: [B, T]
        x, _ = self.rnn(x.view(x.size(0), -1, 1))
        x = torch.hstack((x[:,-1,:self.rnn_hiddens],x[:,0,self.rnn_hiddens:])) if self.bidirectional else x[:,-1,:]
        out, logits = self.fc(x.view(-1,self.fc_inputs))  # sigmoid included
        return out, logits  # [B,]

    
class RNNmodel(nn.Module):
    # classification (transit/non-transit) at each time step
    def __init__(self, rnn_hiddens, fc_hiddens, neg_slope=0, lstm=False, bidirectional=True, num_layers=1, features_in=1):
        super(RNNmodel, self).__init__()
        if lstm:
            self.rnn = nn.LSTM(input_size=features_in, hidden_size=rnn_hiddens, num_layers=num_layers,
                               batch_first=True, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(input_size=features_in, hidden_size=rnn_hiddens, num_layers=num_layers,
                              batch_first=True, bidirectional=bidirectional)
        self.features_in = features_in
        fc_inputs = 2*rnn_hiddens if bidirectional else rnn_hiddens
        self.fc = MLPmodel(fc_inputs, fc_hiddens, neg_slope)

    def forward(self, x, sigmoid=False):
        # input x: [B, T, (F)]
        x, _ = self.rnn(x.view(x.size(0), -1, self.features_in))
        out, logits = self.fc(x)  # sigmoid included
        out, logits = out.view(x.size(0), -1), logits.view(x.size(0), -1)
        return out, logits  # [B, T]
    

class ConfidenceRNNmodel(nn.Module):
    # classification (transit/non-transit) and confidence indication at each time step
    def __init__(self, rnn_hiddens, fc_hiddens, conf_hiddens, neg_slope=0, lstm=False, 
                 bidirectional=True, num_layers=1):
        super(ConfidenceRNNmodel, self).__init__()
        if lstm:
            self.rnn = nn.LSTM(input_size=1, hidden_size=rnn_hiddens, num_layers=num_layers,
                               batch_first=True, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(input_size=1, hidden_size=rnn_hiddens, num_layers=num_layers,
                              batch_first=True, bidirectional=bidirectional)
            
        fc_inputs = 2*rnn_hiddens if bidirectional else rnn_hiddens
        self.fc = MLPmodel(fc_inputs, fc_hiddens, neg_slope)  
        self.conf = MLPmodel(fc_inputs, conf_hiddens, neg_slope)

    def forward(self, x, sigmoid=False):
        # input x: [B, T]
        x, _ = self.rnn(x.view(x.size(0), -1, 1))
        out, logits = self.fc(x)  # sigmoid included
        out, logits = out.view(x.size(0), -1), logits.view(x.size(0), -1)
        conf = self.conf(x)[0].view(x.size(0), -1)  # sigmoid included
        return out, logits, conf  # [B, T], [B, T]
    
    
class RepresentationRNNmodel(nn.Module):
    # classification (transit/non-transit) and representation at each time step
    def __init__(self, rnn_hiddens, fc_hiddens, repr_hiddens, neg_slope=0, lstm=False, 
                 bidirectional=True, num_layers=1):
        super(RepresentationRNNmodel, self).__init__()
        if lstm:
            self.rnn = nn.LSTM(input_size=1, hidden_size=rnn_hiddens, num_layers=num_layers,
                               batch_first=True, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(input_size=1, hidden_size=rnn_hiddens, num_layers=num_layers,
                              batch_first=True, bidirectional=bidirectional)
            
        fc_inputs = 2*rnn_hiddens if bidirectional else rnn_hiddens
        self.fc = MLPmodel(fc_inputs, fc_hiddens, neg_slope)  
        self.repr = MLPmodel(fc_inputs, repr_hiddens, neg_slope, one_out=False)

    def forward(self, x):
        # input x: [B, T]
        x, _ = self.rnn(x.view(x.size(0), -1, 1))
        out, logits = self.fc(x)  # sigmoid included
        out, logits = out.view(x.size(0), -1), logits.view(x.size(0), -1)
        reprs = self.repr(x)[1].view(x.size(0), x.size(1), -1)
        return out, logits, reprs  # [B, T], [B, T, R]

    
class GenerativeRNNmodel(nn.Module):
    # classification (transit/non-transit) and prediction of flux value at each time step
    def __init__(self, rnn_hiddens, fc_hiddens, pred_hiddens, neg_slope=0, lstm=False, 
                 bidirectional=True, num_layers=1):
        super(GenerativeRNNmodel, self).__init__()
        if num_layers > 1:
            print("WARNING: choosing num_layers=1, >1 not implemented")
            num_layers = 1
        if pred_hiddens[-1] != 1:
            print("WARNING: adding pred_hiddens[-1]=1")
            pred_hiddens += [1]
        if lstm:
            self.rnn_cell = nn.LSTMCell(input_size=1, hidden_size=rnn_hiddens)
        else:
            self.rnn_cell = nn.GRUCell(input_size=1, hidden_size=rnn_hiddens)
           
        self.rnn_hiddens = rnn_hiddens
        self.bidirectional = bidirectional
        fc_inputs = 2*rnn_hiddens if bidirectional else rnn_hiddens
        self.fc = MLPmodel(fc_inputs, fc_hiddens, neg_slope)
        self.pred = MLPmodel(rnn_hiddens, pred_hiddens, neg_slope)

    def forward(self, x):
        # input x: [B, T]
        # assuming first and last time step are NOT NaN
        x = x.T.view(-1,x.size(0),1)
        # iterate through light curve in both directions if bidirectional
        x = torch.hstack((x, x.flip(0))) if self.bidirectional else x
        
        nanmsk = torch.isnan(x)
        nans = torch.any(nanmsk, dim=1)
        
        hx = self.rnn_cell(x[0])
        
        rnn_out = [hx]  # outputs same as "regular" rnn
        preds = [x[0], self.pred(hx)[1].view(-1,1)]  # value prediction of next time step
        
        # iterate through light curve
        for i in range(1, x.size(0)):
            if nans[i]:
                # replace nan values with model predictions
                x_in = x[i].clone()
                with torch.no_grad():
                    replace = preds[i][nanmsk[i]]
                    x_in[nanmsk[i]] = replace.clone()
            else:
                x_in = x[i]
                
            hx = self.rnn_cell(x_in, hx)
            rnn_out.append(hx)
            
            if i+1 < x.size(0):
                preds.append(self.pred(hx)[1].view(-1,1))
            
        x = torch.hstack(rnn_out).view(hx.size(0), -1, self.rnn_hiddens)
        
        preds = torch.hstack(preds)
        
        if self.bidirectional:
            # swap time direction of reversed samples
            preds1, preds2 = preds.chunk(2)
            preds = torch.vstack((preds1, preds2.flip(1)))
            x1, x2 = x.chunk(2)
            x = torch.stack((x1, x2.flip(1)), dim=2).view(x1.size(0), x1.size(1), -1)
        
        out, logits = self.fc(x)  # sigmoid included
        out, logits = out.view(x.size(0), -1), logits.view(x.size(0), -1)
        return out, logits, preds  # [B, T], [B(x2), T]
    
    
def num_params(model):
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])
