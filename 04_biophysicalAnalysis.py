#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[1]:


import os
import glob

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl
import seaborn as sns

import torch
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm
import esm


# ### Load data

# In[2]:


# ---------------------------
# Load training data (positives + negatives)
# ---------------------------
csv_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\01_Data_gathering_and_processing\data_2.csv'
train_df = pd.read_csv(csv_path)

# Standardize column names
train_df = train_df.rename(columns={'Sequence': 'sequence'})

# ---------------------------
# Split positives and negatives
# ---------------------------
pos_df = train_df[train_df['Viability'] == 1].copy()
neg_df = train_df[train_df['Viability'] == 0].copy()

pos_df['viability'] = 'positive'
neg_df['viability'] = 'negative'

print(f"Positives loaded: {pos_df.shape}")
print(f"Negatives loaded: {neg_df.shape}")


# ---------------------------
# Keep only required columns and standardize names
# ---------------------------
pos_df = pos_df[['Sequence_ID', 'sequence', 'viability']].rename(
    columns={'Sequence_ID': 'sequence_id'}
)

neg_df = neg_df[['Sequence_ID', 'sequence', 'viability']].rename(
    columns={'Sequence_ID': 'sequence_id'}
)

# ---------------------------
# Convert to format compatible with generated sequences and add tag
# ---------------------------

# For positives
pos_df = pos_df[['sequence_id', 'sequence']].copy()
pos_df['prob_1'] = 1.0
pos_df['pred_label'] = 1
pos_df['tag'] = 'training_positives'

# For negatives
neg_df = neg_df[['sequence_id', 'sequence']].copy()
neg_df['prob_1'] = 0.0
neg_df['pred_label'] = 0
neg_df['tag'] = 'training_negatives'


# In[3]:


import pandas as pd

# Reference sequence
aav2vp1_refSeq = "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"

# Create a DataFrame with the reference row
ref_row = pd.DataFrame({
    "sequence_id": ["reference"],
    "sequence": [aav2vp1_refSeq],
    "prob_1": [1],
    "pred_label": [1],
    # Tag for plotting purposes only; this is not really a training sequence
    "tag": ["training_positives"]
})

# Append to pos_df
pos_df = pd.concat([pos_df, ref_row], ignore_index=True)

# Quick check
pos_df


# In[4]:


# ---------------------------
# Load generated sequences unique (CSV files)
# ---------------------------
dfs = {}

folder_path = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\01_Preliminary sequences analysis"
csv_files = glob.glob(os.path.join(folder_path, "*_unique_sequences.csv"))

for file in csv_files:
    file_name = os.path.splitext(os.path.basename(file))[0]

    df = pd.read_csv(file)

    # Ensure 'sequence_id' exists
    if 'sequence_id' not in df.columns:
        df['sequence_id'] = [f"{file_name}_{i+1}" for i in range(len(df))]

    # Sanity check
    if 'sequence' not in df.columns:
        raise ValueError(f"'sequence' column not found in {file_name}")

    # Check if viability columns exist
    viability_cols = ['prob_1', 'pred_label']
    for col in viability_cols:
        if col not in df.columns:
            # Create empty column if missing
            df[col] = pd.NA

    # Reorder columns: sequence_id, sequence, prob_1, pred_label
    df = df[['sequence_id', 'sequence', 'prob_1', 'pred_label']]

    dfs[file_name] = df

    print(f"Loaded: {file_name} ({df.shape[0]} rows, {df.shape[1]} cols)")


# In[5]:


# ---------------------------
# Add tag to each DataFrame (
# ---------------------------

finetuned_08_df = dfs['finetuned_08_unique_sequences'].copy()
finetuned_08_df['tag'] = 'generated_ft08'

finetuned_12_df = dfs['finetuned_12_unique_sequences'].copy()
finetuned_12_df['tag'] = 'generated_ft12'

nonfinetuned_08_df = dfs['nonfinetuned_08_unique_sequences'].copy()
nonfinetuned_08_df['tag'] = 'generated_nonft08'

nonfinetuned_12_df = dfs['nonfinetuned_12_unique_sequences'].copy()
nonfinetuned_12_df['tag'] = 'generated_nonft12'

reinforced_08_df = dfs['reinforced_08_unique_sequences'].copy()
reinforced_08_df['tag'] = 'generated_reinf08'

reinforced_12_df = dfs['reinforced_12_unique_sequences'].copy()
reinforced_12_df['tag'] = 'generated_reinf12'

# Check counts and tags
print(f"finetuned_08: {len(finetuned_08_df)} rows, tag={finetuned_08_df['tag'].unique()}")
print(f"finetuned_12: {len(finetuned_12_df)} rows, tag={finetuned_12_df['tag'].unique()}")
print(f"nonfinetuned_08: {len(nonfinetuned_08_df)} rows, tag={nonfinetuned_08_df['tag'].unique()}")
print(f"nonfinetuned_12: {len(nonfinetuned_12_df)} rows, tag={nonfinetuned_12_df['tag'].unique()}")
print(f"reinforced_08: {len(reinforced_08_df)} rows, tag={reinforced_08_df['tag'].unique()}")
print(f"reinforced_12: {len(reinforced_12_df)} rows, tag={reinforced_12_df['tag'].unique()}")


# In[6]:


# Join same model type (stack rows)
finetuned_df = pd.concat(
    [finetuned_08_df, finetuned_12_df],
    ignore_index=True
)

nonfinetuned_df = pd.concat(
    [nonfinetuned_08_df, nonfinetuned_12_df],
    ignore_index=True
)

reinforced_df = pd.concat(
    [reinforced_08_df, reinforced_12_df],
    ignore_index=True
)

# Quick check
print("finetuned:", finetuned_df.shape)
print("nonfinetuned:", nonfinetuned_df.shape)
print("reinforced:", reinforced_df.shape)


# In[7]:


## Define the Wimley–White scale

WW_SCALE = {
    "I": -0.81, "L": -0.69, "F": -0.58, "V": -0.53, "M": -0.44,
    "P": -0.31, "W": -0.24, "H": -0.06,
    "T":  0.11, "Q":  0.19, "C":  0.22, "Y":  0.23,
    "A":  0.33, "S":  0.33, "N":  0.43,
    "R":  1.00, "G":  1.14, "E":  1.61, "K":  1.81, "D":  2.41
}

## Function to compute ww
def ww_score(seq):
    values = [WW_SCALE.get(res, 0.0) for res in seq]

    hydrophobic = sum(-v for v in values if v < 0)  # magnitude of negatives
    hydrophilic = sum(v for v in values if v > 0)

    total = hydrophobic + hydrophilic
    return hydrophobic, hydrophilic, total

## Function to compute charge
def compute_charge(seq):
    """
    Returns:
    - total_charge: (#R + #K) - (#D + #E)
    - cationic: #R + #K
    - anionic: #D + #E
    """
    if not isinstance(seq, str):
        return 0, 0, 0

    cationic = seq.count('R') + seq.count('K')
    anionic  = seq.count('D') + seq.count('E')
    total_charge = cationic - anionic

    return total_charge, cationic, anionic


# In[8]:


# Define the anchored-window function
def extract_anchored_window(seq, start_bio=561, c_terminal_keep=147):
    if not isinstance(seq, str):
        return None

    L = len(seq)
    start_idx = start_bio - 1
    end_idx = L - c_terminal_keep

    if end_idx <= start_idx:
        return None

    return seq[start_idx:end_idx]


# In[9]:


## Compute ww and charge for each sequence in each df

# List of all DataFrames
all_dfs = [pos_df, neg_df, finetuned_df, nonfinetuned_df, reinforced_df]

## Loop through each df to extract the window, and comput ww and charge
for df in all_dfs:
    # Extract anchored window
    df["WW_window"] = df["sequence"].apply(extract_anchored_window)
    
    
    # Compute Wimley–White scores
    df[["WW_hydrophobic", "WW_hydrophilic", "WW_total"]] = (
        df["WW_window"].apply(ww_score).apply(pd.Series)
    )
    
    # Compute charges
    df[["total_charge", "cationic", "anionic"]] = (
        df["WW_window"].apply(compute_charge).apply(pd.Series)
    )
    
    # Add combined, simpler columns
    df["hydrophobicity"] = df["WW_hydrophobic"]
    df["hydrophilicity"] = df["WW_hydrophilic"]
    df["cationicity"] = df["cationic"]
    df["anionicity"] = df["anionic"]


# In[10]:


## Ploting

def normalized_hist(data, start, end, step):
    """
    Computes a histogram with normalized counts by number of sequences.
    Returns bin centers and normalized counts.
    """
    bins = np.arange(start, end + step, step)
    hist, edges = np.histogram(data, bins=bins)
    
    hist = hist / len(data)  # normalize
    centers = edges[:-1]     # left edges as bin positions
    
    return centers, hist


# In[11]:


# -----------------------------
# Compute histograms for all datasets
# -----------------------------

datasets = {
    "Training (viable)": pos_df,
    "Training (non-viable)": neg_df,
    "Non-fine-tuned": nonfinetuned_df,
    "Fine-tuned": finetuned_df,
    "Reinforcement": reinforced_df
}

# Dictionary to store ww histogram results
histograms = {}

# Loop over datasets
for name, df in datasets.items():
    histograms[name] = {}
    # Hydrophobic
    histograms[name]["hydrophobic"] = normalized_hist(df["WW_hydrophobic"], 0, 15, 0.2)
    # Hydrophilic
    histograms[name]["hydrophilic"] = normalized_hist(df["WW_hydrophilic"], 0, 50, 0.5)
    # Total
    histograms[name]["total"] = normalized_hist(df["WW_total"], 0, 50, 0.5)

# Dictionary to store charge histogram results
charge_histograms = {}

for name, df in datasets.items():
    charge_histograms[name] = {}
    
    # Total charge
    charge_histograms[name]["total"] = normalized_hist(df["total_charge"], -20, 15, 1)
    # Cationic residues
    charge_histograms[name]["cationic"] = normalized_hist(df["cationic"], -20, 15, 1)
    # Anionic residues
    charge_histograms[name]["anionic"] = normalized_hist(df["anionic"], -20, 15, 1)


# In[12]:


# -----------------------------
# Dataset names and colors
# -----------------------------
dataset_names = [
    "Training (viable)",
    "Training (non-viable)",
    "Non-fine-tuned",
    "Fine-tuned",
    "Reinforcement"
]

colors = {
    "Training (viable)": 'black',
    "Training (non-viable)": 'grey',
    "Non-fine-tuned": 'blue',
    "Fine-tuned": 'green',
    "Reinforcement": 'red'
}

labels = {name: name for name in dataset_names}

# -----------------------------
# Function to annotate reference with diagonal arrow
# -----------------------------
def annotate_reference(ax, ref_value, y_vals, x_vals):
    # Find nearest y in histogram
    nearest_idx = (abs(x_vals - ref_value)).argmin()
    y_arrow = y_vals[nearest_idx]
    
    # Get axis limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    # Fixed diagonal offset (~45 degrees)
    dx = 0.05 * (xmax - xmin)
    dy = 0.05 * (ymax - ymin)
    
    ax.annotate('',
                xy=(ref_value, y_arrow),                       # arrow tip
                xytext=(ref_value - dx, y_arrow - dy),         # arrow tail (diagonal)
                arrowprops=dict(arrowstyle='-|>',              # fixed head style
                                lw=1,                          # line width
                                color='black',
                                mutation_scale=15),            # fixed head size
                ha='center')

# -----------------------------
# Create 2x3 grid
# -----------------------------
fig, axs = plt.subplots(2, 3, figsize=(20,13), gridspec_kw={'hspace': 0.45, 'wspace': 0.25})

# -----------------------------
# Row 1: Wimley–White scores
# -----------------------------
ww_types = ["hydrophobic", "hydrophilic", "total"]
ww_x_limits = {
    "hydrophobic": (0, 15),
    "hydrophilic": (5, 40),
    "total": (5, 40)
}

for i, score_type in enumerate(ww_types):
    ax = axs[0, i]
    for ds_name in dataset_names:
        x, y = histograms[ds_name][score_type]
        ax.plot(x, y, color=colors[ds_name], label=labels[ds_name])
    
    # Annotate reference
    ref_value = pos_df.loc[pos_df['sequence_id']=='reference', f'WW_{score_type}'].values[0]
    x_vals, y_vals = histograms["Training (viable)"][score_type]
    annotate_reference(ax, ref_value, y_vals, x_vals)
    
    ax.set_title(score_type.capitalize(), fontsize=18)
    ax.set_xlabel("WW score", fontsize=16)
    if i == 0:
        ax.set_ylabel("Normalized frequency", fontsize=16)
        ax.legend(fontsize=12)
    ax.set_xlim(*ww_x_limits[score_type])
    ax.tick_params(labelsize=14)

# -----------------------------
# Row 2: Charges
# -----------------------------
charge_types = ["cationic", "anionic", "total"]
charge_x_limits = {
    "cationic": (-5, 15),
    "anionic": (-5, 15),
    "total": (-15, 15)
}

for i, score_type in enumerate(charge_types):
    ax = axs[1, i]
    for ds_name in dataset_names:
        x, y = charge_histograms[ds_name][score_type]
        ax.plot(x, y, color=colors[ds_name])
    
    # Annotate reference
    ref_value = pos_df.loc[pos_df['sequence_id']=='reference',
                           'total_charge' if score_type=='total' else score_type].values[0]
    x_vals, y_vals = charge_histograms["Training (viable)"][score_type]
    annotate_reference(ax, ref_value, y_vals, x_vals)
    
    ax.set_title(score_type.capitalize(), fontsize=18)
    ax.set_xlabel("Residue count / net charge", fontsize=16)
    if i == 0:
        ax.set_ylabel("Normalized frequency", fontsize=16)
    ax.set_xlim(*charge_x_limits[score_type])
    ax.tick_params(labelsize=14)

# -----------------------------
# Save figure
# -----------------------------
save_path = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\03_Biophysical_analysis\ww_charge_plots_3.png"

# Save the current figure
fig.savefig(save_path, dpi=600, bbox_inches='tight')

print(f"Figure saved to: {save_path}")
print("")

plt.show()


# In[13]:


# -----------------------------
# Dataset names and colors
# -----------------------------
dataset_names = [
    "Training (viable)",
    "Training (non-viable)",
    "Non-fine-tuned",
    "Fine-tuned",
    "Reinforcement"
]

colors = {
    "Training (viable)": 'black',
    "Training (non-viable)": 'grey',
    "Non-fine-tuned": 'blue',
    "Fine-tuned": 'green',
    "Reinforcement": 'red'
}

labels = {name: name for name in dataset_names}

# -----------------------------
# Function to annotate reference with diagonal arrow
# -----------------------------
def annotate_reference(ax, ref_value, y_vals, x_vals):
    nearest_idx = (abs(x_vals - ref_value)).argmin()
    y_arrow = y_vals[nearest_idx]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx = 0.05 * (xmax - xmin)
    dy = 0.05 * (ymax - ymin)
    ax.annotate('',
                xy=(ref_value, y_arrow),
                xytext=(ref_value - dx, y_arrow - dy),
                arrowprops=dict(arrowstyle='-|>', lw=1, color='black', mutation_scale=15),
                ha='center')

# -----------------------------
# Create 1x4 grid
# -----------------------------
fig, axs = plt.subplots(1, 4, figsize=(20,4), gridspec_kw={'wspace': 0.2})

# -----------------------------
# Columns: hydrophobic, hydrophilic, cationic, anionic
# -----------------------------
columns = ["hydrophobic", "hydrophilic", "cationic", "anionic"]
x_limits = {
    "hydrophobic": (0, 15),
    "hydrophilic": (5, 40),
    "cationic": (-5, 15),
    "anionic": (-5, 15)
}

# Formatter for y-axis with 2 decimals
y_formatter = mticker.FormatStrFormatter('%.2f')

for i, score_type in enumerate(columns):
    ax = axs[i]
    for ds_name in dataset_names:
        if score_type in ["hydrophobic", "hydrophilic"]:
            x, y = histograms[ds_name][score_type]
        else:
            x, y = charge_histograms[ds_name][score_type]
        ax.plot(x, y, color=colors[ds_name], label=labels[ds_name])
    
    # Annotate reference
    if score_type in ["hydrophobic", "hydrophilic"]:
        ref_value = pos_df.loc[pos_df['sequence_id']=='reference', f'WW_{score_type}'].values[0]
        x_vals, y_vals = histograms["Training (viable)"][score_type]
    else:
        ref_value = pos_df.loc[pos_df['sequence_id']=='reference', score_type].values[0]
        x_vals, y_vals = charge_histograms["Training (viable)"][score_type]
    annotate_reference(ax, ref_value, y_vals, x_vals)
    
    ax.set_title(score_type.capitalize(), fontsize=16)
    ax.set_xlabel("Score / Residue count", fontsize=16)
    if i == 0:
        ax.set_ylabel("Normalized frequency", fontsize=16)
        ax.legend(title='Sequence source', fontsize=12, title_fontsize=14)
    
    ax.set_xlim(*x_limits[score_type])
    ax.tick_params(axis='both', labelsize=14)
    
    # Apply y-axis formatter
    ax.yaxis.set_major_formatter(y_formatter)

# -----------------------------
# Save figure
# -----------------------------
save_path = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\03_Biophysical_analysis\ww_charge_plots_1row_3.png"
fig.savefig(save_path, dpi=600, bbox_inches='tight')
print(f"Figure saved to: {save_path}\n")

plt.show()


# In[14]:


# -----------------------------
# Clean DataFrame (drop duplicates)
# -----------------------------
cols_to_drop = ['hydrophobicity', 'hydrophilicity', 'cationicity', 'anionicity']
reinforced_df = reinforced_df.drop(columns=cols_to_drop, errors='ignore')

# -----------------------------
# Add reference sequence
# -----------------------------
aav2vp1_refSeq = "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"

ref_df = pd.DataFrame({
    'sequence_id': ['reference'],
    'sequence': [aav2vp1_refSeq],
    'prob_1': [1.0],
    'pred_label': [1],
    'tag': ['reference']
})

reinforced_df = pd.concat([reinforced_df, ref_df], ignore_index=True)

# Inspect
reinforced_df


# In[15]:


# -----------------------------
# Export all datasets to CSV
# -----------------------------

# Training datasets
pos_df.to_csv("pos_df_with_biophysicals_3.csv", index=False)
neg_df.to_csv("neg_df_with_biophysicals_3.csv", index=False)

# Generated datasets
nonfinetuned_df.to_csv("nonfinetuned_df_with_biophysicals_3.csv", index=False)
finetuned_df.to_csv("finetuned_df_with_biophysicals_3.csv", index=False)
reinforced_df.to_csv("reinforced_df_with_biophysicals_3.csv", index=False)


# In[16]:


## Generate embeddings & calculate distances
# -----------------------------
# Embedding functions 
# -----------------------------
@torch.no_grad()
def embed_batch(batch_records, model, alphabet, batch_converter, device):
    labels, strs, tokens = batch_converter(batch_records)
    tokens = tokens.to(device)
    out = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
    representations = out["representations"][model.num_layers]
    cls_batch, mean_batch = {}, {}
    for i, seq_id in enumerate(labels):
        token_ids = tokens[i]
        non_pad_mask = token_ids != alphabet.padding_idx
        non_pad_indices = non_pad_mask.nonzero(as_tuple=True)[0]
        residue_indices = non_pad_indices[1:-1]  # exclude CLS and EOS
        per_residue = representations[i, residue_indices, :].cpu()
        cls_batch[seq_id] = representations[i, 0, :].cpu()
        mean_batch[seq_id] = per_residue.mean(dim=0)
    return cls_batch, mean_batch

def add_esm_embeddings_cpu(df, seq_column="sequence", batch_size=16):
    model_name = "esm2_t33_650M_UR50D"
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    batch_converter = alphabet.get_batch_converter()

    df = df.copy()
    df["_tmp_id"] = df.index.astype(str)
    records = list(zip(df["_tmp_id"].tolist(), df[seq_column].tolist()))

    cls_embeddings = {}

    for i in tqdm(range(0, len(records), batch_size)):
        batch = records[i:i+batch_size]
        cls_b, _ = embed_batch(batch, model, alphabet, batch_converter, device)
        cls_embeddings.update(cls_b)

    df["cls_ESM2_emb"] = df["_tmp_id"].map(cls_embeddings)
    df.drop(columns="_tmp_id", inplace=True)
    return df

def tensor_or_list_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, list):
        return np.array([t.item() if isinstance(t, torch.Tensor) else t for t in x])
    else:
        return np.array(x)

# -----------------------------
# Setup
# -----------------------------
batch_size = 1000
output_dir = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\03_Biophysical_analysis\reinforced_batches"
os.makedirs(output_dir, exist_ok=True)

# Make sure reference is in dataframe
if 'reference' not in reinforced_df['sequence_id'].values:
    aav2vp1_refSeq = "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"
    ref_df = pd.DataFrame({
        'sequence_id': ['reference'],
        'sequence': [aav2vp1_refSeq],
        'prob_1': [1.0],
        'pred_label': [1],
        'tag': ['reference']
    })
    reinforced_df = pd.concat([reinforced_df, ref_df], ignore_index=True)

# -----------------------------
# Reference embedding
# -----------------------------
ref_row = reinforced_df[reinforced_df['sequence_id'] == 'reference']
ref_row = add_esm_embeddings_cpu(ref_row, batch_size=1)
ref_emb = tensor_or_list_to_numpy(ref_row["cls_ESM2_emb"].iloc[0]).reshape(1, -1)

# -----------------------------
# Process in batches
# -----------------------------
N = len(reinforced_df)
n_batches = int(np.ceil(N / batch_size))
for i in range(n_batches):
    batch_file = os.path.join(output_dir, f"reinforced_df_{i:03d}.csv")
    if os.path.exists(batch_file):
        print(f"Batch {i} already exists. Skipping...")
        continue

    start = i * batch_size
    end = min((i + 1) * batch_size, N)
    batch_df = reinforced_df.iloc[start:end].copy()

    # Generate embeddings
    batch_df = add_esm_embeddings_cpu(batch_df, batch_size=16)

    # Compute distance
    X = np.vstack(batch_df["cls_ESM2_emb"].apply(tensor_or_list_to_numpy).values)
    batch_df["dist_to_reference"] = cosine_distances(X, ref_emb).ravel()

    # Drop embeddings to save memory
    batch_df.drop(columns=["cls_ESM2_emb"], inplace=True)

    # Save batch
    batch_df.to_csv(batch_file, index=False)
    print(f"Saved batch {i}: {batch_file}")

# -----------------------------
# Reconstruct full DataFrame
# -----------------------------
batch_files = sorted(glob.glob(os.path.join(output_dir, "reinforced_df_*.csv")))
reinforced_df_final = pd.concat([pd.read_csv(f) for f in batch_files], ignore_index=True)

print("All batches processed and merged.")

#Inspect
reinforced_df_final


# In[17]:


reinforced_df_final.to_csv("reinforced_df_with_biophysicals_and_distances_3.csv", index=False)


# In[18]:


reinforced_df_final


# In[27]:


# --------------------------------------------------
# Filter reinforced_df_final positives
# AND with high probability
# --------------------------------------------------
reinforced_df_final_positives = reinforced_df_final[
    (reinforced_df_final['prob_1'] > 0.5)
].copy()

print(f"Sequences in reinforced_df_final with prob_1 > 0.5: {len(safe_df)}")


# In[19]:


# --------------------------------------------------
# Compute safe region based on 90% of positives
# --------------------------------------------------
hydro_bounds = np.percentile(pos_df['WW_total'], [5, 95])
charge_bounds = np.percentile(pos_df['total_charge'], [5, 95])

print("Hydrophobicity bounds (safe region):", hydro_bounds)
print("Charge bounds (safe region):", charge_bounds)

# --------------------------------------------------
# Filter reinforced_df_final for sequences inside safe region
# AND with high probability
# --------------------------------------------------
safe_df = reinforced_df_final[
    (reinforced_df_final['WW_total'] >= hydro_bounds[0]) &
    (reinforced_df_final['WW_total'] <= hydro_bounds[1]) &
    (reinforced_df_final['total_charge'] >= charge_bounds[0]) &
    (reinforced_df_final['total_charge'] <= charge_bounds[1]) &
    (reinforced_df_final['prob_1'] > 0.5)
].copy()

print(f"Sequences inside safe region with prob_1 > 0.5: {len(safe_df)}")


# In[20]:


max_dist_reinforced = reinforced_df_final['dist_to_reference'].max()
max_dist_safe = safe_df['dist_to_reference'].max()

print("Maximum distance to reference in reinforced_df:", max_dist_reinforced)
print("Maximum distance to reference in safe_df:", max_dist_safe)


# In[21]:


safe_df 


# In[30]:


# --------------------------------------------------
# Define 4x4 grid based on safe_df
# --------------------------------------------------
n_bins = 5
hydro_bins = np.linspace(safe_df['WW_total'].min(), safe_df['WW_total'].max(), n_bins + 1)
charge_bins = np.linspace(safe_df['total_charge'].min(), safe_df['total_charge'].max(), n_bins + 1)

safe_df['hydro_bin'] = pd.cut(safe_df['WW_total'], bins=hydro_bins, labels=False, include_lowest=True)
safe_df['charge_bin'] = pd.cut(safe_df['total_charge'], bins=charge_bins, labels=False, include_lowest=True)

# --------------------------------------------------
# Select top sequence per grid cell (highest dist_to_reference)
# --------------------------------------------------
grid_selection = []
for i in range(n_bins):
    for j in range(n_bins):
        cell_df = safe_df[(safe_df['hydro_bin'] == i) & (safe_df['charge_bin'] == j)]
        if len(cell_df) > 0:
            top_idx = cell_df['dist_to_reference'].idxmax()
            grid_selection.append(top_idx)

grid_selection_df = safe_df.loc[grid_selection].copy()
grid_selection_df = grid_selection_df.reset_index(drop=True)

print(f"Sequences selected from grid: {len(grid_selection_df)}")

# --------------------------------------------------
# Plotting
# --------------------------------------------------

sns.set(style="ticks", context="talk")  # keep axis ticks visible

# -----------------------------
# Define font sizes
# -----------------------------
axis_label_size = 14
title_size = 14
tick_label_size = 12
legend_size = 12
colorbar_label_size = 12
colorbar_tick_size = 12

fig, ax = plt.subplots(figsize=(8, 6))

# -----------------------------
# Colormap and normalization
# -----------------------------
cmap = mpl.cm.Greys  # higher distance = darker
min_val = safe_df['dist_to_reference'].min()
max_val = safe_df['dist_to_reference'].max()
norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)

# -----------------------------
# Scatter all sequences in safe region
# -----------------------------
sc = ax.scatter(
    safe_df['total_charge'],
    safe_df['WW_total'],
    c=safe_df['dist_to_reference'],
    cmap=cmap,
    norm=norm,
    alpha=0.7
    #label='Sequences in 90% (positives) region'
)

# -----------------------------
# Highlight selected sequences with red edge
# -----------------------------
ax.scatter(
    grid_selection_df['total_charge'],
    grid_selection_df['WW_total'],
    c=grid_selection_df['dist_to_reference'],
    cmap=cmap,
    norm=norm,
    s=120,
    edgecolors='red',   # red outline
    linewidths=1.5,
    label='Selected sequences'
)

# -----------------------------
# Reference point
# -----------------------------
ref_charge = -2
ref_hydro = 20.33
ax.scatter(ref_charge, ref_hydro, c='black', s=150, marker='X', label='Reference')

# -----------------------------
# Draw 4x4 grid lines
# -----------------------------
for x in charge_bins:
    ax.axvline(x, color='blue', linestyle='--', alpha=0.3, zorder=0)
for y in hydro_bins:
    ax.axhline(y, color='blue', linestyle='--', alpha=0.3, zorder=0)

# -----------------------------
# Labels and title with font size
# -----------------------------
ax.set_xlabel("Charge (total residues count charge)", fontsize=axis_label_size)
ax.set_ylabel("Hydrophobicity (total WW score)", fontsize=axis_label_size)
ax.set_title("Sequence Selection Across Charge & Hydrophobicity Grid", fontsize=title_size)

# -----------------------------
# Colorbar
# -----------------------------
cbar = plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.1)
cbar.set_label('Novelty', fontsize=colorbar_label_size)
cbar.ax.tick_params(labelsize=colorbar_tick_size)

# -----------------------------
# Legend outside the plot
# -----------------------------
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=legend_size)

# -----------------------------
# Minor ticks and tick labels font size
# -----------------------------
ax.minorticks_on()
ax.tick_params(axis='both', which='major', direction='in', length=6, labelsize=tick_label_size)
ax.tick_params(axis='both', which='minor', direction='in', length=3)

plt.tight_layout()

# Save figure at 600 dpi
save_path = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\03_Biophysical_analysis\sequence_selection_plot_3.png"
plt.savefig(save_path, dpi=600, bbox_inches='tight')

plt.show()


# In[32]:


# Create a DataFrame with the 9 selected sequences
selected_sequences_df = grid_selection_df.copy()

# Reset index
selected_sequences_df = selected_sequences_df.reset_index(drop=True)

#Export 
selected_sequences_df.to_csv("final_sequences_to_produce_4.csv", index=False)

# Display
selected_sequences_df


# In[ ]:




