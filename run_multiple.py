
from main import main

datasets = ["mrpc"] #["rte", "cola", "qqp"]#["sst2small", "mrpcsmall", "mnlismall", "qnlismall",,"cola", "qnli","mnli"]#[ ]#,"mnli"]
split_by = ["layer"]#"layer","qkv",
n_opts = [1]
models = ["bert"]#, "roberta"]
update_rule = ["cycle"]#"cycle",  "impact_mag"
optim = [ "adamsls"]#,"adam", "sgdsls"]#, "sgd", "sgdsls"]"adam", 
combine = [0]
numexp = 5
batch_size = [32]
cs = [0.3]
betas = [0.99]
only_grad_smoothing = [False]

def create_config(name, ds, split, n_opt, model, opt, update_r = "cycle", i = 0, combine = 0, batch_size = 32, c = 0.1, cls = "transformer", beta = 0.99, onlygradsmooth = False):
    with open(name, 'w') as f:
            f.write("[DEFAULT]\n")
            f.write("batch_size = "+ str(batch_size) +"\n")
            f.write("checkpoint = None\n")
            f.write("directory = results/"  + ds + opt + str(n_opt) + model + split + update_r +str(combine)+ str(i) + "\n")
            f.write("seed = 42\n")
            f.write("epochs = 5\n")
            f.write("dataset = " + ds + "\n")
            f.write("optim = " + opt + "\n")
            f.write("num_diff_opt =" + str(n_opt) + "\n")
            f.write("model = " + model + "\n")
            f.write("split_by = " + split + "\n")
            f.write("update_rule = " + update_r + "\n")
            f.write("combine = " + str(combine) + "\n")
            f.write("c = " + str(c) + "\n")
            f.write("cls = " + cls + "\n")
            f.write("type = " + "NLP" + "\n")
            f.write("beta = "+ str(beta)+ "\n")
            f.write("onlygradientsmoothing = "+ str(onlygradsmooth)+ "\n")
   # print("results/"  +ds + opt+ str(n_opt) + model + split )
    main(name)

for ds in datasets:
    for model in models:
        for opt in optim:
            if "sls" in opt:
                for smooth in only_grad_smoothing:
                    for beta in betas:
                        for update_r in update_rule:
                            for split in split_by:
                                if split == "layer":
                                    for n_opt in n_opts:
                                        for comb in combine:
                                            for bs in batch_size:
                                                for c in cs:
                                                    for i in range(numexp):
                                                        create_config("config_gen.json", ds, split, n_opt, model , opt, 
                                                            update_r, i,combine = comb, batch_size = bs, c = c, beta = beta, onlygradsmooth = smooth)
                                else:
                                    for i in range(numexp):
                                        create_config("config_gen.json", ds, split, 1, model , opt, update_r, i, beta = beta, onlygradsmooth = smooth)
            else:
                for i in range(numexp):
                    create_config("config_gen.json", ds, "layer", 1, model , opt,"cycle", i)


            
            
            
            
            
            
            
           
            
            


