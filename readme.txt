#Author: Joshua Topper

The goal of this script is to determine the taxonomic diversity of 
bacteria present in the microbiome of 50 samples. 

The following file should be in you taxome_diversity directory:
scriptDiS.py

The folder inputfiles should be in you taxome_diversity directory.
In the inputfiles folder:
clnical_data.txt - contains information pertaining to 50 samples
distanceFiles - folder that contains the distance values for each sample
diversityScores - folder that contains the diversity scores for each sample

To execute the script make sure you are inside the taxome_diversity directory and type 
"python3 scriptDiS.py" and hit enter. This will:

1) Output an appended clinical data file called clinical_data.stats.txt that
contains the mean values and standard deviations of the sample's diversity scores

2) Make scatter plots for the two highest average diversity scores the lowest diversity 
score (3 plots total). It will also make elbow method plots for each scatter plot.

*NOTE: All outputs will end up in the taxome_diversity directory.
