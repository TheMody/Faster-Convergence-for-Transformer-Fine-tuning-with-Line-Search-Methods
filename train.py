import torch

import numpy as np
import random
from embedder import NLP_embedder
from data import load_data, SimpleDataset, load_wiki, load_wikiandbook
# Fixing seed for reproducibility
SEED = 999
random.seed(SEED)
np.random.seed(SEED)

ACTIVE_METRIC_NAME = 'accuracy'
REWARD_ATTR_NAME = 'objective'
datasets = [ "mnli","cola", "sst2", "mrpc","qqp", "rte"]#"qqp", "rte" 
eval_ds = [ "rtesmall", "qqpsmall","qqp", "rte"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
            

def train(args, config):
    max_epochs = int(config["DEFAULT"]["epochs"])

    batch_size = int(config["DEFAULT"]["batch_size"])

    dataset = config["DEFAULT"]["dataset"]
    optimizer = config["DEFAULT"]["optim"]


    print("dataset:", dataset)
    lr = 2e-5

    args.number_of_diff_lrs = int(config["DEFAULT"]["num_diff_opt"])
    args.opts = {"lr": lr, "opt": optimizer}
    args.combine = float(config["DEFAULT"]["combine"])
    args.ds = dataset
    args.beta = float(config["DEFAULT"]["beta"])
    args.split_by = config["DEFAULT"]["split_by"]
    args.update_rule = config["DEFAULT"]["update_rule"]
    args.model = config["DEFAULT"]["model"]
    args.savepth = config["DEFAULT"]["directory"]
    args.c = float(config["DEFAULT"]["c"])
    args.o_grad_smooth = config["DEFAULT"]["onlygradientsmoothing"]=="True"
    num_classes = 2
    if "mnli" in dataset:
        num_classes = 3

    if args.train_type == "cls":
        print("loading model")
        model = NLP_embedder(num_classes = num_classes,batch_size = batch_size,args =  args)

        model = model.to(device)
        print("loading dataset")
        X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(name=dataset)
        print("training model on dataset", dataset)
        model.fit(X_train, Y_train, epochs=max_epochs, X_val= X_val, Y_val = Y_val)
        accuracy = model.evaluate(X_val,Y_val).item()
        print("acuraccy on ds:", accuracy)

    if args.train_type == "mlm":
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        args.opts["lr"] = 1e-4
        model = NLP_embedder(num_classes = tokenizer.vocab_size,batch_size = batch_size,args =  args).to(device)
        print("loading dataset")
        ds = load_wikiandbook(batch_size)
        model.fitmlm(ds, 2000000, config["DEFAULT"]["directory"]+"/model.pt")
       