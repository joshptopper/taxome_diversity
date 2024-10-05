#Author: Joshua Topper

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler

#get the directory of the script
script_directory = os.path.dirname(os.path.abspath(__file__))

#define paths to input files and folders
input_folder = os.path.join(script_directory, "inputfiles")
clinical_data_file = os.path.join(input_folder, "clinical_data.txt")
diversity_scores_folder = os.path.join(input_folder, "diversityScores")
distance_files_folder = os.path.join(input_folder, "distanceFiles")

#read in the clinical data file
clinical_data = pd.read_csv(clinical_data_file, sep="\t")

#function to calculate mean and standard deviation for diversity scores
def calculate_stats(diversity_scores):
    diversity_scores = np.array(diversity_scores)
    avg = np.mean(diversity_scores)
    std = np.std(diversity_scores)
    return avg, std

#initialize new lists inside the loop
for index, row in clinical_data.iterrows():
    #initialize empty lists for each iteration
    averages = []
    std_devs = []

    diversity_scores_file = os.path.join(diversity_scores_folder, f"{row['code_name']}.diversity.txt")
    #read diversity scores for the current sample
    diversity_scores = np.loadtxt(diversity_scores_file)
    #calculate mean and standard deviation
    avg, std = calculate_stats(diversity_scores)
    #append to lists
    averages.append(avg)
    std_devs.append(std)

    #add new columns to the clinical data dataframe
    clinical_data.loc[index, 'averages'] = avg
    clinical_data.loc[index, 'std'] = std

#save updated dataframe to a new file
output_file = os.path.join(script_directory, "clinical_data.stats.txt")
clinical_data.to_csv(output_file, sep="\t", index=False)


#plot scatter plot and save as png file
def plot_and_save_scatter(animal):
    distance_file = os.path.join(distance_files_folder, f"{animal}.distance.txt")
    distances = pd.read_csv(distance_file, header=None).values
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=distances[:,0], y=distances[:,1])
    plt.title(f"Distance Scatter Plot for {animal}")
    plt.xlabel("Distance 1")
    plt.ylabel("Distance 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(script_directory, f"{animal}_scatter.png"))
    plt.close()

#plot and save scatter plots for animals with highest and lowest average diversity scores
animal_with_highest_avg = clinical_data.nlargest(2, 'averages')['code_name'].tolist() #(adjust numeric values as needed)
animal_with_lowest_avg = clinical_data.nsmallest(1, 'averages')['code_name'].tolist() #(adjust numeric values as needed)
animals_to_plot = animal_with_highest_avg + animal_with_lowest_avg

for animal in animals_to_plot:
    plot_and_save_scatter(animal)

#load the scatter plot data
scatter_data = []
for animal in animals_to_plot:
    distance_file = os.path.join(distance_files_folder, f"{animal}.distance.txt")
    distances = pd.read_csv(distance_file, header=None).values
    scatter_data.append(distances)


num_clusters_range = range(1, 11)

#perform K-means clustering for each scatter plot
for idx, data in enumerate(scatter_data):
    animal_name = animals_to_plot[idx]
    inertia = []

    plt.figure(figsize=(12, 6))
    for num_clusters in num_clusters_range:
        centroids, distortion = kmeans(data, num_clusters)
        inertia.append(distortion)  #use the distortion value from kmeans

    #plot the elbow method
    plt.plot(num_clusters_range, inertia, marker='o')
    plt.title(f"Elbow Method for {animal_name}")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.xticks(num_clusters_range)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(script_directory, f"elbow_method_{animal_name}.png"))
    plt.show()