
from main import main

datasets = [ "tiny-imagenet"]#"tiny-imagenet"]  # ["rte", "cola", "qqp"]#["sst2small", "mrpcsmall", "mnlismall", "qnlismall","sst2","mrpc" ,"cola", "qnli","mnli"]#[ ]#,"mnli"]
split_by = ["layer"]#"layer","qkv",
n_opts = [1]
models = [ "effNet"]#, "roberta"]
update_rule = ["cycle"]#"cycle",  "impact_mag"
optim = [ "adamsls","adam", "oladamsls", "sgd"]#["oladamsls", "adamsls","adam"]#, "sgd", "sgdsls"]"adam", 
combine = [ 0]
numexp = 5
batch_size = [32]
cs = [ 0.3]
epochs = [20]
clss = ["cnn"]
n_hidden =[2]
betas = [0.99]

def create_config(name, ds, split, n_opt, model, opt, update_r = "cycle", i = 0, combine = 0, batch_size = 32, c = 0.1, epochs = 5, cls = "cnn", n_hid = 1, beta = 0.99):
    with open(name, 'w') as f:
            f.write("[DEFAULT]\n")
            f.write("batch_size = "+ str(batch_size) +"\n")
            f.write("checkpoint = None\n")
            f.write("directory = results/"  + ds + opt + str(n_opt) + model + split + update_r +str(combine)+ str(i) + "\n")
            f.write("seed = 42\n")
            f.write("epochs = "+str(epochs)+"\n")
            f.write("dataset = " + ds + "\n")
            f.write("optim = " + opt + "\n")
            f.write("num_diff_opt =" + str(n_opt) + "\n")
            f.write("model = " + model + "\n")
            f.write("split_by = " + split + "\n")
            f.write("update_rule = " + update_r + "\n")
            f.write("combine = " + str(combine) + "\n")
            f.write("c = " + str(c) + "\n")
            f.write("cls = " + cls + "\n")
            f.write("type = " + "img" + "\n")
            f.write("num_hidden_dims = " + str(n_hid) + "\n")
            f.write("beta = " + str(beta) + "\n")
            
   # print("results/"  +ds + opt+ str(n_opt) + model + split )
    main(name)

for ds in datasets:
    for model in models:
        for opt in optim:
            for e in epochs:
                for n_hid in n_hidden:
                    for cls in clss:
                        for bs in batch_size:
                            if "sls" in opt:
                                for beta in betas:
                                    for update_r in update_rule:
                                        for split in split_by:
                                            if split == "layer":
                                                for n_opt in n_opts:
                                                    for comb in combine:
                                                            for c in cs:
                                                                for i in range(numexp):
                                                                    create_config("config_gen.json", ds, split, n_opt, model , opt, update_r, i,combine = comb, batch_size = bs, c = c,n_hid= n_hid, epochs = e, cls = cls, beta= beta)
                                            else:
                                                for i in range(numexp):
                                                    create_config("config_gen.json", ds, split, 1, model , opt, update_r, i,  batch_size = bs, epochs = e, cls = cls,n_hid= n_hid, beta= beta)
                            else:
                                for i in range(numexp):
                                    create_config("config_gen.json", ds, "layer", 1, model , opt,"cycle", i,  batch_size = bs, epochs = e, cls = cls,n_hid= n_hid)



            
            
            
            
            
            
            
           
            
            


