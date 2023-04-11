from transformers import AutoFeatureExtractor, ResNetModel, ResNetConfig
from datasets import load_dataset
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import numpy as np
from copy import deepcopy
from data import load_wiki
from sls.adam_sls import AdamSLS
from sls.slsoriginal.adam_sls import AdamSLS as olAdamSLS
import wandb
from cosine_scheduler import CosineWarmupScheduler
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Dense_classifier(nn.Module):

    def __init__(self, input_dim,hidden_dim, num_classes, batch_size, args):
        super(Dense_classifier, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.args = args
        self.num_hidden_dims = args.hidden_dims
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.list_of_modules = nn.ModuleList()
        for n in range(self.num_hidden_dims):          
            self.list_of_modules.append(nn.Linear(hidden_dim,hidden_dim))
        self.fc4 = nn.Linear(hidden_dim,self.num_classes)
        self.criterion = nn.CrossEntropyLoss()

        if args.opts["opt"] == "adam":    
            self.optimizer = optim.Adam(self.parameters(), lr=args.opts["lr"] )
        if args.opts["opt"] == "sgd":    
            self.optimizer = optim.SGD(self.parameters(), lr=args.opts["lr"] )
        if args.opts["opt"] == "oladamsls":    
            self.optimizer = AdamSLS( [[param for name,param in self.named_parameters() if not "pooler" in name]] , c = 0.1, smooth = False )
        if args.opts["opt"] == "adamsls":    
            self.optimizer = AdamSLS( [[param for name,param in self.named_parameters() if not "pooler" in name]] ,strategy = args.update_rule, combine_threshold = args.combine, c = self.args.c )
        if args.opts["opt"] == "sgdsls":    
            self.optimizer = AdamSLS( [[param for name,param in self.named_parameters() if not "pooler" in name]],strategy = args.update_rule, combine_threshold = args.combine, base_opt = "scalar",gv_option = "scalar", c = self.args.c  )

    def forward(self,x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        for module in self.list_of_modules:   
            x = F.relu(module(x))
        x = self.fc4(x)
        return x

    def fit(self,data, epochs, eval_ds = None):
        wandb.init(project="SLSforDifferentLayersImage"+self.args.ds, name = self.args.split_by + "_" + self.args.opts["opt"] + "_" + self.args.model +
        "_" + str(self.args.number_of_diff_lrs) +"_"+ self.args.savepth, entity="pkenneweg", 
        group = "dense_"+self.args.split_by + "_" + self.args.opts["opt"] + "_" + self.args.model +"_" + 
        str(self.args.number_of_diff_lrs) + self.args.update_rule + str(self.args.combine)+"_c"+ str(self.args.c)+ "n_hid_"+ str(self.args.hidden_dims) )
        
        if not "sls" in self.args.opts["opt"]:
            self.scheduler = CosineWarmupScheduler(optimizer= self.optimizer, 
                                                warmup = math.ceil(len(data)*epochs *0.1) ,
                                                    max_iters = math.ceil(len(data)*epochs ))

        accuracy = None
        accsteps = 0
        accloss = 0
        for e in range(epochs):
            for index in range(len(data)):
                startsteptime = time.time()
                batch_x, batch_y = next(iter(data))
                # print(batch_y)
                # print(self(batch_x))

                if "sls" in self.args.opts["opt"]:
                    closure = lambda : self.criterion(self(batch_x), batch_y)
                    self.optimizer.zero_grad()
                    loss = self.optimizer.step(closure = closure)
                else:
                    self.optimizer.zero_grad()
                    y_pred = self(batch_x)

                    loss = self.criterion(y_pred, batch_y)    
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()      

                dict = {"loss": loss.item() , "time_per_step":time.time()-startsteptime}    
                if "sls" in  self.args.opts["opt"]:
                    for a,step_size in enumerate( self.optimizer.state['step_sizes']):
                        dict["step_size"+str(a)] = step_size
                        dict["avg_grad_norm"+str(a)] = self.optimizer.state["grad_norm_avg"][a]
                        dict["loss_decrease"+str(a)] = self.optimizer.state["loss_dec_avg"][a]
                else:
                    dict["step_size"+str(0)] = self.scheduler.get_last_lr()[0]
                wandb.log(dict)
                accloss = accloss + loss.item()
                accsteps += 1
                if index % np.max((1,int((len(data))*0.1))) == 0:
                    print(index, accloss/ accsteps)
                    accsteps = 0
                    accloss = 0
            if not eval_ds == None:
                accuracy = self.evaluate(eval_ds)
                print("accuracy at epoch", e, accuracy)
                wandb.log({"accuracy": accuracy})
        wandb.finish()
        return accuracy

    @torch.no_grad()
    def evaluate(self, data):
        acc = 0
        for _ in range(len(data)):
            batch_x, batch_y = next(iter(data))
            batch_x = self(batch_x)
            y_pred = torch.argmax(batch_x, dim = 1)
            accuracy = torch.sum(batch_y == y_pred)
            acc += accuracy
        
        acc = acc.item()/(len(data)*self.batch_size)
        return acc


