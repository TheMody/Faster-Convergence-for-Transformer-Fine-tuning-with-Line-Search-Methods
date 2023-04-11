import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import numpy as np
from transformers.utils import logging
from transformers import glue_convert_examples_to_features, DataCollatorForLanguageModeling
from copy import deepcopy
from torch.autograd import variable
from torch.utils.data import DataLoader
from data import load_wiki
import os
from sls.adam_sls import AdamSLS
from sls.slsoriginal.adam_sls import AdamSLS as orAdamSLS
import wandb
from cosine_scheduler import CosineWarmupScheduler
logging.set_verbosity_error()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





    


class NLP_embedder(nn.Module):

    def __init__(self,  num_classes, batch_size, args, mode = "mlm"):
        super(NLP_embedder, self).__init__()
        self.type = 'nn'
        self.batch_size = batch_size
        self.padding = True
        self.bag = False
        self.num_classes = num_classes
        self.lasthiddenstate = 0
        self.args = args
        self.mode = mode
        self.loss = nn.CrossEntropyLoss()

        if args.model == "bert":
            from transformers import BertTokenizer, BertModel
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.output_length = 768
        if args.model == "roberta":
            from transformers import RobertaTokenizer, RobertaModel
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.model = RobertaModel.from_pretrained('roberta-base')
            self.output_length = 768

        
        self.fc1 = nn.Linear(self.output_length,self.num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        

        if args.split_by == "layer":
            if args.number_of_diff_lrs > 1:
                pparamalist = []
                for i in range(args.number_of_diff_lrs):
                    paramlist = []
                    optrangelower = math.ceil((12.0/(args.number_of_diff_lrs-2)) *(i-1))
                    optrangeupper = math.ceil((12.0/(args.number_of_diff_lrs-2)) * (i))
                    
                    optrange = list(range(optrangelower,optrangeupper))
                    if i == 0 or i == args.number_of_diff_lrs-1:
                        optrange =[]
                    for name,param in self.named_parameters():
                        if "encoder.layer." in name:
                            included = False
                            for number in optrange:
                                if "." +str(number)+"." in name:
                                    included = True
                            if included:
                                paramlist.append(param)
                              #  print("included", name , "in", i)
                        else:
                            if "embeddings." in name:
                                if i == 0:
                                    paramlist.append(param)
                                 #   print("included", name , "in", i)
                            else:
                                if i == args.number_of_diff_lrs-1 and not "pooler" in name:
                                    paramlist.append(param)
                                  #  print("included", name , "in", i)
                                  #  print(name, param.requires_grad, param.grad)
                    pparamalist.append(paramlist)
                if args.opts["opt"] == "adamsls":  
                    self.optimizer = AdamSLS(pparamalist,strategy = args.update_rule , combine_threshold = args.combine, c = self.args.c, o_grad_smooth=self.args.o_grad_smooth)
                if args.opts["opt"] == "sgdsls":  
                    self.optimizer = AdamSLS( pparamalist,strategy = args.update_rule, combine_threshold = args.combine, base_opt = "scalar",gv_option = "scalar" , c = self.args.c, o_grad_smooth=self.args.o_grad_smooth)
                 #   self.optimizer.append(SgdSLS(pparamalist ))
            else:
                if args.opts["opt"] == "adam":    
                    self.optimizer = optim.Adam(self.parameters(), lr=args.opts["lr"] )
                if args.opts["opt"] == "radam":    
                    self.optimizer = optim.RAdam(self.parameters(), lr=args.opts["lr"] )
                if args.opts["opt"] == "sgd":    
                    self.optimizer = optim.SGD(self.parameters(), lr=args.opts["lr"] )
                if args.opts["opt"] == "adamsls":    
                    self.optimizer = AdamSLS( [[param for name,param in self.named_parameters() if not "pooler" in name]] , c = self.args.c, beta_s = self.args.beta, o_grad_smooth=self.args.o_grad_smooth)
                if args.opts["opt"] == "oladamsls":    
                    self.optimizer = AdamSLS( [[param for name,param in self.named_parameters() if not "pooler" in name]] , c = self.args.c, smooth=False)
                if args.opts["opt"] == "amsgradsls":    
                    self.optimizer = orAdamSLS( [param for name,param in self.named_parameters() if not "pooler" in name] ,base_opt = "amsgrad", c = 0.1)
                if args.opts["opt"] == "sgdsls":    
                    self.optimizer = AdamSLS( [[param for name,param in self.named_parameters() if not "pooler" in name]], base_opt = "scalar",gv_option = "scalar", c = self.args.c , beta_s = self.args.beta, o_grad_smooth=self.args.o_grad_smooth)
        else:
            querylist = []
            keylist = []
            valuelist = []
            elselist = [] 
            for name,param in self.named_parameters():
                if not "pooler" in name:
                    if "query" in name:
                        querylist.append(param)
                    else:
                        if "key" in name:
                            keylist.append(param)
                        else:
                            if "value" in name:
                                valuelist.append(param)
                            else:
                                elselist.append(param)
            if args.opts["opt"] == "adamsls":    
                self.optimizer = AdamSLS( [keylist,querylist,valuelist,elselist],strategy = args.update_rule, combine_threshold = args.combine)

            
        
    def forward(self, x_in):
        x = self.model(**x_in).last_hidden_state
        if self.mode == "cls":
            x = x[:, self.lasthiddenstate]
        x = self.fc1(x)
        return x
    
    
     
    def fit(self, x, y, epochs=1, X_val= None,Y_val= None):
        wandb.init(project="SLSforDifferentLayers"+self.args.ds, name = self.args.split_by + "_" + self.args.opts["opt"] + "_" + self.args.model +
            "_" + str(self.args.number_of_diff_lrs) +"_"+ self.args.savepth, entity="pkenneweg", 
            group = "avgarmijo_momentum_"+self.args.split_by + "_" + self.args.opts["opt"] + "_" + self.args.model +"_" + str(self.args.number_of_diff_lrs) + self.args.update_rule 
            + str(self.args.combine)+"bs"+ str(self.batch_size) +"c"+ str(self.args.c)+"beta"+ str(self.args.beta)+"only_grad_smooth"+ str(self.args.o_grad_smooth))
        #wandb.watch(self)
        
        self.mode = "cls"
        if (not "sls" in  self.args.opts["opt"]):
            self.scheduler= CosineWarmupScheduler(optimizer= self.optimizer, 
                                                warmup = math.ceil(len(x)*epochs *0.1 / self.batch_size) ,
                                                    max_iters = math.ceil(len(x)*epochs  / self.batch_size))
        


        accuracy = None
        accsteps = 0
        accloss = 0
        for e in range(epochs):
            start = time.time()
            for i in range(math.ceil(len(x) / self.batch_size)):
                startsteptime = time.time()
                ul = min((i+1) * self.batch_size, len(x))
                batch_x = x[i*self.batch_size: ul]
                batch_y = y[i*self.batch_size: ul]
                batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 256, truncation = True)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                batch_y = batch_y.to(device)
                batch_x = batch_x.to(device)

                if "sls" in  self.args.opts["opt"]:
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

                if (i*self.batch_size) % 32 == 0:
                    dict = {"loss": loss.item() , "time_per_step":time.time()-startsteptime}#, "backtracks": np.sum(self.optimizer[a].state['n_backtr'][-1] for a in range(len(self.optimizer)))}
                    if "sls" in  self.args.opts["opt"]:
                        for a,step_size in enumerate( self.optimizer.state['step_sizes']):
                            dict["step_size"+str(a)] = step_size
                            dict["avg_grad_norm"+str(a)] = self.optimizer.state["grad_norm_avg"][a]
                            dict["loss_decrease"+str(a)] = self.optimizer.state["loss_dec_avg"][a]
                    else:
                        dict["step_size"+str(0)] = self.scheduler.get_last_lr()[0]
                        #      print(dict["step_size"+str(a)])
                    wandb.log(dict)#,  step=i*self.batch_size +  e*len(x))
                    accloss = accloss + loss.item()
                    accsteps += 1
                    if i % np.max((1,int((len(x)/self.batch_size)*0.1))) == 0:
                        print(i, accloss/ accsteps)
                        accsteps = 0
                        accloss = 0

            if X_val != None:
                with torch.no_grad():
                    accuracy = self.evaluate(X_val, Y_val).item()
                    print("accuracy after", e, "epochs:",accuracy, "time per epoch", time.time()-start)
                    wandb.log({"accuracy": accuracy})#,  step=i*self.batch_size +  e*len(x))
            else:
                print("epoch",e,"time per epoch", time.time()-start)
                
                
        wandb.finish()
        return accuracy
        
    @torch.no_grad()
    def evaluate(self, X,Y):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Y = Y.to(device)
        y_pred = self.predict(X)
        accuracy = torch.sum(Y == y_pred)
        accuracy = accuracy/Y.shape[0]
        return accuracy


    def mask(self, input_ids, mask_token_id = 103):
        # create random array of floats in equal dimension to input_ids
        rand = torch.rand(input_ids.shape).to(device)
        # where the random array is less than 0.15, we set true
        mask_arr = (rand < 0.15) * (input_ids != 101) * (input_ids != 102)* (input_ids != 0)
        mask_arr1 = (rand < 0.15*0.8)* (input_ids != 101) * (input_ids != 102)* (input_ids != 0) #* (mask_arr)
        mask_arr2 = (0.15*0.8 < rand)* (rand < 0.15*0.9)* (input_ids != 101) * (input_ids != 102)* (input_ids != 0)
      #  mask_arr3 = (0.15*0.9 < rand < 0.15)* (input_ids != 101) * (input_ids != 102)* (input_ids != 0) not needed since just nothing is done

        input_ids = torch.where(mask_arr1, mask_token_id, input_ids)#80% normal masking
        input_ids = torch.where(mask_arr2, (torch.rand(1, device = device)* self.tokenizer.vocab_size).long(), input_ids)#10% random token
        #last 10% is just original token and does not need to be replaced
        return input_ids, mask_arr
    
    def compute_mlm_loss(self,output,labels):
        output = output.flatten(start_dim = 0, end_dim = 1)
        labels = labels.flatten(start_dim = 0, end_dim = 1)
        loss = self.loss(output, labels)
        return loss

    #todo test mlm results

    def fitmlm(self,dataset, steps, checkpoint_pth = None):
        if (not self.args.opts["opt"] == "adamsls") and (not self.args.opts["opt"] == "sgdsls"):
            self.scheduler =[]
            for i in range(len(self.optimizer)): 
                self.scheduler.append(CosineWarmupScheduler(optimizer= self.optimizer[i], 
                                                warmup = 3000 ,
                                                    max_iters = steps))
        wandb.init(project="mlmwithSLS", name = self.args.split_by + "_" + self.args.opts["opt"] + "_" + self.args.model +
        "_" + str(self.args.number_of_diff_lrs) +"_"+ self.args.savepth)
        wandb.watch(self)
        self.mode = "mlm"
        accsteps = 0
        accloss = 0
        for i,data in enumerate(dataset):
            startsteptime = time.time()
            input = self.tokenizer(data, return_tensors="pt", padding=True, max_length = 256, truncation = True)
            input = input.to(device)
            labels = torch.clone(input["input_ids"])
            input["input_ids"], mask = self.mask(input["input_ids"])
            labels = torch.where(mask, labels, -100)

            if self.args.opts["opt"] == "adamsls" or self.args.opts["opt"] == "sgdsls":
                closure = lambda : self.compute_mlm_loss(self(input),labels)

                for a in range(len(self.optimizer)):
                    self.optimizer[a].zero_grad()

                for a in range(len(self.optimizer)):
                    loss = self.optimizer[a].step(closure = closure)
            else:
                for a in range(len(self.optimizer)):
                    self.optimizer[a].zero_grad()

                loss = self.compute_mlm_loss(self(input),labels)  
                loss.backward()
                for a in range(len(self.optimizer)):
                    self.optimizer[a].step()
                for a in range(len(self.scheduler)):
                    self.scheduler[a].step()  

            dict = {"loss": loss.item() , "time_per_step":time.time()-startsteptime}
            if self.args.opts["opt"] == "adamsls" or self.args.opts["opt"] == "sgdsls":
                for a,step_size in enumerate( self.optimizer[0].state['step_sizes']):
                        dict["step_size"+str(a)] = step_size
            else:
                for a,scheduler in enumerate( self.scheduler):
                    dict["step_size"+str(a)] = scheduler.get_last_lr()[0]
            wandb.log(dict)
            accloss = accloss + loss.item()
            accsteps += 1
            if i % 100 == 0:
                print(i, accloss/ accsteps)
                accsteps = 0
                accloss = 0
            if not checkpoint_pth == None and i % np.max((1,int(steps*0.1))) == 0:
                torch.save(self, checkpoint_pth)
            if i >= steps:
                break
    
    def predict(self, x):
        resultx = None

        for i in range(math.ceil(len(x) / self.batch_size)):
            ul = min((i+1) * self.batch_size, len(x))
            batch_x = x[i*self.batch_size: ul]
            batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 256, truncation = True)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch_x = batch_x.to(device)
            batch_x = self(batch_x)
            if resultx is None:
                resultx = batch_x.detach()
            else:
                resultx = torch.cat((resultx, batch_x.detach()))

     #   resultx = resultx.detach()
        return torch.argmax(resultx, dim = 1)
    
    def embed(self, x):
        resultx = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(math.ceil(len(x) / self.batch_size)):
            ul = min((i+1) * self.batch_size, len(x))
            batch_x = x[i*self.batch_size: ul]
            batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 256, truncation = True)
            batch_x = batch_x.to(device)
            batch_x = self.model(**batch_x,output_hidden_states = True)   
            batch_x = batch_x.hidden_states[-1]
            batch_x = batch_x[:, self.lasthiddenstate]
            if resultx is None:
                resultx = batch_x.detach()
            else:
                resultx = torch.cat((resultx, batch_x.detach()))

     #   resultx = resultx.detach()
        return resultx
    
    

        
        

