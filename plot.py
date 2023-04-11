

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd 
import wandb



def smoothing(list, length = 20):
    return [np.mean(list[max(i-length, 0 ): i]) for i in range(len(list))]

def exp_smoothing(list, beta = 0.99):
    smoothed = []
    for i in range(len(list)):
        if i == 0:
            smoothed.append(list[i])
        else:
            smoothed.append(beta*smoothed[i-1] + (1-beta)*list[i])
    return smoothed

def decrease_per_epoch(list, epochs = 5):
    length = len(list)
    print(length)
    decrease_list = [list[int(i*length/epochs)] - list[int((i+1)*length/epochs)] for i in range(epochs)]
    return decrease_list
    
if __name__ == '__main__': 
    api = wandb.Api()
    entity, project = "pkenneweg", "SLSforDifferentLayersmnli"  # set to your entity and project 
    runs = api.runs(entity + "/" + project) 
    prev_group = ""
    acc_loss = []
    acc_accuracy = []
    acc_steps = []
    i = -1
    colortable = [(1,0,0), (0,0,1),   (0,1,0),(0.1,0.1,0.1)]
   # colortable = list(colors.TABLEAU_COLORS.values())

    def drawaccumulation(acc_loss,acc_steps, i):
         if len(acc_loss) > 4:
            minlen = min([len(a) for a in acc_loss])
            print(minlen)
            acc_loss = [a[:minlen] for a in acc_loss]
            acc_steps = [a[:minlen] for a in acc_steps]
            acc_loss = np.asarray([smoothing(a) for a in acc_loss])
            acc_steps = np.mean(np.asarray(acc_steps), axis = 0).astype(int)
            mean = np.mean(acc_loss, axis = 0)
            std = np.std(acc_loss, axis = 0)
            error = std/np.sqrt(len(acc_loss))
            c = list(colors.to_rgba(colortable[i]))
            c[3] = c[3]*0.3
            c = tuple(c)
            plt.fill_between(acc_steps, mean + error, mean - error, color = c)
            plt.plot(acc_steps,mean, color = colortable[i])
            
            

            
   # print(colortable)
    legendlist = []
    for run in runs:
        if run.state == "finished":
            if "visible" in run.tags:
              #  if "cycle" in run.name or "impact_mag" in run.name:
                    print(run.name)
                    if not run.group == prev_group:
                        if not i == -1:
                          #  drawaccumulation(acc_loss,acc_steps, i)
                           # legendlist.append( prev_group)
                            acc_loss = np.asarray(acc_loss)
                            acc_loss_mean = np.mean(acc_loss, axis = 0)
                            acc_loss_mean = exp_smoothing(acc_loss_mean)
                            decreases = decrease_per_epoch(acc_loss_mean)
                            print(decreases)
                            decreases = np.log(decreases)
                            print("log",decreases)
                            print("mean",np.mean(decreases))
                         #   print(run.group, np.mean([np.mean(a) for a in acc_loss]))
                        acc_loss = []
                        acc_accuracy = []
                        acc_steps =[]
                        prev_group = run.group
                        i = i +1
                        
                    
                    hist = run.scan_history(keys=["loss","_step"])
                    histl = np.asarray([row["loss"] for row in hist])
                  #  hists = [row["_step"] for row in hist]
                    # hist = run.history()
                    # histl = hist["loss"]
                    # hists = hist["_step"]
                    # hists = hists[histl.isnull() == False]
                    # histl = histl[histl.isnull() == False]

                   # print(len(hists))
                   # print(len(histl))
                   # print(acc_loss)
                    acc_loss.append(histl)
                   # acc_steps.append(hists)
                 #   acc_accuracy.append( run.history()["accuracy"][run.history()["accuracy"].isnull() == False].to_numpy()) 
    drawaccumulation(acc_loss,acc_steps, i)
    legendlist.append( prev_group)
    print(legendlist)
    legendlist = ["SGDSLS", "ADAM","PLASLS", "ADAMSLS"]
    plt.legend(legendlist)
    plt.xlabel("step")
    plt.xlim(0,53000)
    plt.ylabel("loss")
    plt.yscale("log")
    plt.grid(visible=True, axis = "y",linestyle= "--")
    plt.savefig("plot.png", dpi = 1000)
    plt.show()   
































