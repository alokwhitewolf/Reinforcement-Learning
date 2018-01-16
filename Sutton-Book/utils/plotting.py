from matplotlib import pyplot as plt
import numpy as np

def plot_gridworld_values(values):
    dims = values.shape[0]
    
    #plt.axis('off')
    to_plot = values
    to_plot = np.around(to_plot, decimals=2).tolist()
    
    plt.figure(figsize=(10,7))
    plt.imshow(to_plot)
    plt.title('')
    for (j,i),label in np.ndenumerate(to_plot):
        plt.text(i,j,label,ha='center',va='center',color="white")
        plt.text(i,j,label,ha='center',va='center',color="white")
    plt.axis('off')

def plot_gridworld_policy(values):
    dims = values.shape[0]
    
    #plt.axis('off')
    to_plot = values
    to_plot = np.around(to_plot, decimals=2).tolist()
    
    plt.figure(figsize=(10,7))
    plt.imshow(to_plot)
    plt.title('')
    for (j,i),label in np.ndenumerate(to_plot):
        plt.text(i,j,label,ha='center',va='center',color="white")
        plt.text(i,j,label,ha='center',va='center',color="white")

    for i in range(dims):
        for j in range(dims):
            
            hash_table = {}
            
            try:
                if j is not 0:
                    hash_table['left'] = to_plot[i][j-1]
            except:
                pass
            
            try:
                if j is not dims:
                    hash_table['right'] = to_plot[i][j+1]
            except:
                pass
            
            try:
                if i is not 0:
                    hash_table['up'] = to_plot[i-1][j]
            except:
                pass
            
            try:
                if i is not dims:
                    hash_table['down'] = to_plot[i+1][j]
            except:
                pass
           
            global optimal_action 
            optimal_action = []
            best_neighbor = -1000
            
            for key, value in hash_table.items():
                
                if value>best_neighbor:
                    optimal_action = [key]
                    best_neighbor = value
                elif value==best_neighbor:
                    optimal_action.append(key)
            
            directions = {'up':[0,-1],'down':[0,1],'left':[-1,0],'right':[1,0]}
            for action in optimal_action:
                plt.arrow(j, i,directions[action][0]*.375, directions[action][1]*.375, alpha = 0.5, width = 0.015, edgecolor = 'red', facecolor = 'red', lw = 2, zorder = 5)
            plt.axis('off')