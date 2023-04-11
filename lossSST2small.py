

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd 
import wandb



def smoothing(list, length = 10):
    return [np.mean(list[max(i-length, 0 ): i]) for i in range(len(list))]
    
if __name__ == '__main__': 
    api = wandb.Api()
    entity, project = "pkenneweg", "SLSforDifferentLayerssst2small"  # set to your entity and project 
    runs = api.runs(entity + "/" + project) 
    prev_group = ""
    acc_loss = []
    acc_accuracy = []
    i = -1
    colortable = [(1,0,0),(0,1,0),  (0,0,1), (0.1,0.1,0.1)]
   # colortable = list(colors.TABLEAU_COLORS.values())

    def drawaccumulation(acc_loss, i):
         if len(acc_loss) > 4:
            acc_loss = [smoothing(a) for a in acc_loss]
            mean = np.mean(acc_loss, axis = 0)
            std = np.std(acc_loss, axis = 0)
            error = std/np.sqrt(len(acc_loss))
            c = list(colors.to_rgba(colortable[i]))
            c[3] = c[3]*0.3
            c = tuple(c)
            plt.fill_between(np.arange(len(mean)), mean + error, mean - error, color = c)
            plt.plot(mean, color = colortable[i])
            
            

            
   # print(colortable)
    legendlist = []
    for run in runs:
        if run.state == "finished":
            if "visible" in run.tags:
              #  if "cycle" in run.name or "impact_mag" in run.name:
                    print(run.name)
                    if not run.group == prev_group:
                        if not i == -1:
                            drawaccumulation(acc_loss, i)
                            legendlist.append( prev_group)
                        acc_loss = []
                        acc_accuracy = []
                        prev_group = run.group
                        i = i +1
                    acc_loss.append(run.history()["loss"][run.history()["loss"].isnull() == False].to_numpy())
                    acc_accuracy.append( run.history()["accuracy"][run.history()["accuracy"].isnull() == False].to_numpy()) 
    drawaccumulation(acc_loss, i)
    legendlist.append( prev_group)
    print(legendlist)
    legendlist = ["SGDSLS", "PLASLS", "ADAM", "ADAMSLS"]
   # plt.legend(legendlist)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.savefig("plot.png", dpi = 1000)
    plt.show()   
































