# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 18:29:51 2025

@author: JoanaCatarino
"""
import os 
import tifffile 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Map results file from injection segmentation in Napari to annotation volume(Voxels'value in striatum parcellation)
    # Use voxel value to search for structure information in structure_str.csv

# Define animal and cohort
animal_id = 889722
cohort = 2

# Import .csv file with injection site segmentation data 
results_df = pd.read_csv(f'L:/dmclab/Joana/Tracing/Tlx3/Cohort {cohort}/{animal_id}/results/injection_site/green/{animal_id}_injection_site.csv')

# Create a copy of the file and keep only the columns that we will use to extract the voxel information
results_voxel = results_df[['ap_coords', 'dv_coords', 'ml_coords']].copy()

# Import the annotation volume
annotation_str = tifffile.imread('L:/dmclab/Joana/Tracing/Injection_analysis/parcel_brain.tiff')
print(annotation_str.shape) # Should be (1320, 800, 1140)

# Import .csv file with the voxel values for the str parcellation
str_tree = pd.read_csv('L:/dmclab/Joana/Tracing/Injection_analysis/structures_str.csv')

# Create a dictioary to associate each voxel value with the correct name based on the str_tree struture
id_map = {k:v for k,v in zip(str_tree.id, str_tree.name)}
print(id_map)

# Apply the map to the results file
all_voxel = results_voxel.values
all_voxel = all_voxel.T # transpose

new_results_df = results_df.copy()
new_results_df['str_id'] = annotation_str[all_voxel[0], all_voxel[1], all_voxel[2]]
new_results_df['str_name'] = new_results_df['str_id'].map(id_map)
new_results_df['str_name'].fillna('other regions', inplace=True)

# Most brains will have injections in both hemispheres so let's make the left hemisphere(check this) 
#  with negative voxel numbers to distinguish them
new_results_df.loc[new_results_df.ml_mm<0, 'str_id'] *= -1

# Save new .csv file
new_results_df.to_csv(f'L:/dmclab/Joana/Tracing/Tlx3/Cohort {cohort}/{animal_id}/injection_results/{animal_id}_inj_results.csv')

#%%

# Plots

## Pie Chart plots

# Load the data
df = new_results_df

# Filter positive and negative str_id values
df_pos = df[df["ml_mm"] > 0]
df_neg = df[df["ml_mm"] < 0]

# Combine unique structure names across both groups
unique_structures = pd.Index(df_pos["str_name"].unique()).union(df_neg["str_name"].unique())

# Define color for other regions
other_regions_color = '#C0A4CB'

# Generate palette for all structures *excluding* "other regions"
structures_no_other = [s for s in unique_structures if s != "other regions"]
palette = sns.color_palette("crest", n_colors=len(structures_no_other))

color_map = dict(zip(structures_no_other, palette))
color_map["other regions"] = other_regions_color

# Value counts
pos_counts = df_pos["str_name"].value_counts(normalize=True) * 100
neg_counts = df_neg["str_name"].value_counts(normalize=True) * 100

# Get the correct color per slice
pos_colors = [color_map[name] for name in pos_counts.index]
neg_colors = [color_map[name] for name in neg_counts.index]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Positive pie chart
axes[0].pie(pos_counts, colors=pos_colors, autopct='%1.1f%%', startangle=140)
axes[0].set_title("Positive str_id Values")

# Negative pie chart
axes[1].pie(neg_counts, colors=neg_colors, autopct='%1.1f%%', startangle=140)
axes[1].set_title("Negative str_id Values")

# Shared legend
legend_labels = list(color_map.keys())
legend_colors = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                            markerfacecolor=color_map[label], markersize=10)
                 for label in legend_labels]

fig.legend(legend_colors, legend_labels,
           title="Structures",
           loc='lower center',
           bbox_to_anchor=(0.5, -0.05),
           ncol=4,
           frameon=False)

fig.suptitle(f'Injection site distribution - mouse {animal_id} - Cohort {cohort}')
plt.tight_layout()
plt.subplots_adjust(top=0.85, bottom=0.15)

fig.savefig(f'L:/dmclab/Joana/Tracing/Tlx3/Cohort {cohort}/{animal_id}/injection_results/{animal_id}_inj_charts.png', dpi=300, bbox_inches='tight')


## Histogram plots

# Load the data
df = new_results_df

# Split the data by positive and negative str_id values
df_pos = df[df['str_id'] > 0]
df_neg = df[df['str_id'] < 0]

# Plot histograms side by side
fig2 = plt.figure(figsize=(14,6))

# Positive str_id
plt.subplot(1, 2, 1)
plt.hist(df_pos['ap_mm'], bins=30, edgecolor='black', color='skyblue')
plt.xlabel('AP (mm)')
plt.ylabel('Count')
plt.title('AP (mm) Distribution for Positive str_id')
plt.grid(True)

# Negative str_id
plt.subplot(1, 2, 2)
plt.hist(df_neg['ap_mm'], bins=30, edgecolor='black', color='#75BA96')
plt.xlabel('AP (mm)')
plt.ylabel('Count')
plt.title('AP (mm) Distribution for Negative str_id')
plt.grid(True)

fig2.suptitle(f'Injection site distribution - mouse {animal_id} - Cohort {cohort}')
plt.tight_layout()

fig2.savefig(f'L:/dmclab/Joana/Tracing/Tlx3/Cohort {cohort}/{animal_id}/injection_results/{animal_id}_inj_histo.png', dpi=300, bbox_inches='tight')


