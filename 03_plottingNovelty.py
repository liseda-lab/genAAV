#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_distances
from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from tqdm import tqdm
import torch
import esm


# In[2]:


SAMPLE_SIZE=500


# In[3]:


# ---------------------------
# Load training data
# ---------------------------
csv_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\01_Data_gathering_and_processing\data_2.csv'
train_df = pd.read_csv(csv_path)

# Standardize column names
train_df = train_df.rename(columns={'Sequence': 'sequence',
                                    'Sequence_ID': 'sequence_id',
                                    'Viability': 'viability'})

print(f"Total sequences loaded: {train_df.shape}")

# ---------------------------
# Sample sequences from full dataset
# ---------------------------
train_df = train_df.sample(
    n=min(SAMPLE_SIZE, len(train_df)),
    random_state=42
).reset_index(drop=True)

print(f"Sampled {len(train_df)} sequences.")

# ---------------------------
# Keep required columns
# ---------------------------
train_df = train_df[['sequence_id', 'sequence', 'viability']]

# ---------------------------
# Convert to required format
# ---------------------------
train_formatted_df = train_df[['sequence_id', 'sequence']].copy()

# Probability and predicted label directly from viability
train_formatted_df['prob_1'] = train_df['viability'].astype(float)
train_formatted_df['pred_label'] = train_df['viability'].astype(int)

# Single tag for all training data
train_formatted_df['tag'] = 'training_data'

# ---------------------------
# Check final format
# ---------------------------
print(train_formatted_df.head())
print(train_formatted_df.tail())
print(train_formatted_df.shape)
print(train_formatted_df['pred_label'].value_counts())


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
# Combine 08 and 12 for each strategy, sample, and add tag
# ---------------------------

# Finetuned
finetuned_combined = pd.concat([
    dfs['finetuned_08_unique_sequences'],
    dfs['finetuned_12_unique_sequences']
], ignore_index=True)

finetuned_df = finetuned_combined.sample(
    n=min(SAMPLE_SIZE, len(finetuned_combined)),
    random_state=42
).reset_index(drop=True)

finetuned_df['tag'] = 'generated_finetuned'


# Non-finetuned
nonfinetuned_combined = pd.concat([
    dfs['nonfinetuned_08_unique_sequences'],
    dfs['nonfinetuned_12_unique_sequences']
], ignore_index=True)

nonfinetuned_df = nonfinetuned_combined.sample(
    n=min(SAMPLE_SIZE, len(nonfinetuned_combined)),
    random_state=42
).reset_index(drop=True)

nonfinetuned_df['tag'] = 'generated_nonfinetuned'


# Reinforced
reinforced_combined = pd.concat([
    dfs['reinforced_08_unique_sequences'],
    dfs['reinforced_12_unique_sequences']
], ignore_index=True)

reinforced_df = reinforced_combined.sample(
    n=min(SAMPLE_SIZE, len(reinforced_combined)),
    random_state=42
).reset_index(drop=True)

reinforced_df['tag'] = 'generated_reinforced'


# ---------------------------
# Check counts and tags
# ---------------------------
print(f"finetuned: {len(finetuned_df)} rows, tag={finetuned_df['tag'].unique()}")
print(f"nonfinetuned: {len(nonfinetuned_df)} rows, tag={nonfinetuned_df['tag'].unique()}")
print(f"reinforced: {len(reinforced_df)} rows, tag={reinforced_df['tag'].unique()}")


# In[ ]:


# List all DataFrames to combine
all_dfs = [
    train_formatted_df,
    finetuned_df,
    nonfinetuned_df,
    reinforced_df
]

# Concatenate into one unified DataFrame
combined_df = pd.concat(all_dfs, ignore_index=True)

# ---------------------------
# Add reference sequence
# ---------------------------

# reference sequence
aav2vp1_refSeq = "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"

# Create a DataFrame for the reference
ref_df = pd.DataFrame({
    'sequence_id': ['reference'],
    'sequence': [aav2vp1_refSeq],
    'prob_1': [1.0],
    'pred_label': [1],
    'tag' : 'reference'
})

# Append to the combined DataFrame
combined_df = pd.concat([combined_df, ref_df], ignore_index=True)



# In[8]:


# -----------------------------
# Function to embed a batch of sequences
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
        cls_embedding = representations[i, 0, :].cpu()
        mean_embedding = per_residue.mean(dim=0)
        
        cls_batch[seq_id] = cls_embedding
        mean_batch[seq_id] = mean_embedding

    return cls_batch, mean_batch

# -----------------------------
# Compute embeddings for a DataFrame in-place (CPU-only)
# -----------------------------
def add_esm_embeddings_cpu(df, seq_column="Sequence", batch_size=16):
    """
    Adds CLS and mean embeddings to a DataFrame (CPU-only).
    df: pandas DataFrame with protein sequences
    seq_column: column containing the sequences
    batch_size: number of sequences to process per batch
    """
    # Load pretrained ESM2 model
    model_name = "esm2_t33_650M_UR50D"
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    device = torch.device("cpu")  # CPU only
    model.to(device)
    batch_converter = alphabet.get_batch_converter()

    # Use DataFrame index as IDs
    df = df.copy()
    df["_tmp_id"] = df.index.astype(str)
    records = list(zip(df["_tmp_id"].tolist(), df[seq_column].tolist()))

    cls_embeddings, mean_embeddings = {}, {}

    # Process in batches
    for i in tqdm(range(0, len(records), batch_size)):
        batch = records[i:i+batch_size]
        cls_b, mean_b = embed_batch(batch, model, alphabet, batch_converter, device)
        cls_embeddings.update(cls_b)
        mean_embeddings.update(mean_b)

    # Add embeddings to DataFrame
    df["cls_ESM2_emb"] = df["_tmp_id"].map(cls_embeddings)
    df["aa_ESM2_emb"] = df["_tmp_id"].map(mean_embeddings)
    df.drop(columns="_tmp_id", inplace=True)

    return df


# In[ ]:


# Compute embeddings 
# ---------------------------
print("Generating embeddings...")
combined_df = add_esm_embeddings_cpu(combined_df, seq_column="sequence", batch_size=16)


# In[ ]:


def tensor_or_list_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, list):
        # list of tensors
        return np.array([t.item() if isinstance(t, torch.Tensor) else t for t in x])
    else:
        return np.array(x)

combined_df["cls_ESM2_emb"] = combined_df["cls_ESM2_emb"].apply(tensor_or_list_to_numpy)


# In[ ]:


## Calculate cosine distance
# ---------------------------
# Stack all embeddings into a 2D array
# ---------------------------
X = np.vstack(combined_df['cls_ESM2_emb'].apply(np.array).values)  # shape (N, 1280)

# ---------------------------
# Extract reference embedding and make it 2D
# ---------------------------
ref_emb = np.array(
    combined_df.loc[combined_df['sequence_id'] == 'reference', 'cls_ESM2_emb'].values[0]
).reshape(1, -1)  # shape (1, 1024)

# ---------------------------
# Compute cosine distance to reference
# ---------------------------
combined_df['dist_to_reference'] = cosine_distances(X, ref_emb).ravel()


# In[ ]:


# Define output folder (change path if needed)
output_folder =  r'C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\01_Preliminary sequences analysis'
combined_df.to_csv(f"{output_folder}/combined_test.csv", index=False)


# In[63]:


# ---------------------------
# Prepare data for plotting
# ---------------------------

# Filter out the reference sequence
plot_df = combined_df[combined_df['tag'] != 'reference']

# Desired plotting order
tag_order = [
    'training_data',
    'generated_nonfinetuned',
    'generated_finetuned',
    'generated_reinforced'
]

# Pretty display names
tag_labels = {
    'training_data': 'Training',
    'generated_nonfinetuned': 'Non-fine-tuned',
    'generated_finetuned': 'Fine-tuned',
    'generated_reinforced': 'Reinforced'
}

# Keep only relevant tags and enforce order
plot_df = plot_df[plot_df['tag'].isin(tag_order)]
plot_df['tag'] = pd.Categorical(
    plot_df['tag'],
    categories=tag_order,
    ordered=True
)

# ---------------------------
# Output path
# ---------------------------
output_dir = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\02_Novelty_score"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "distance_to_reference_by_tag_violin.png"
)

# ---------------------------
# Violin + strip plot (centered points, legend inside)
# ---------------------------
plt.figure(figsize=(7, 5))

# Violin (white fill + black edge)
sns.violinplot(
    data=plot_df,
    x='tag',
    y='dist_to_reference',
    order=tag_order,
    color='white',
    inner='quartile',
    linewidth=1.5,      # thickness of the edge
    saturation=1,
    edgecolor='black'   # FORCE black edge
)

# Overlay points colored by pred_label, centered
sns.stripplot(
    data=plot_df,
    x='tag',
    y='dist_to_reference',
    order=tag_order,
    hue='pred_label',
    palette={1: 'green', 0: 'red'},
    size=5,
    jitter=0.2,          # slight horizontal jitter
    dodge=False,         # keep points centered
    alpha=0.7
)

plt.ylabel("Distance to Reference", fontsize=14)
plt.xlabel("Sequence Tag", fontsize=14)
plt.title("Distribution of Distance to Reference", fontsize=16)

# Pretty x-axis labels (larger font)
plt.xticks(
    ticks=range(len(tag_order)),
    labels=[tag_labels[t] for t in tag_order],
    rotation=0,
    fontsize=12
)

# ---------------------------
# Legend inside plotting area (top left)
# ---------------------------
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    handles,
    ["Non-viable", "Viable"],
    title="Predicted viability",
    frameon=True,
    framealpha=0.9,
    loc='upper left',
    fontsize=10,
    title_fontsize=10
)

plt.tight_layout()

# ---------------------------
# Save & show
# ---------------------------
plt.savefig(output_path, dpi=600, bbox_inches="tight")
plt.show()


# In[70]:


## t-SNE plotting (sequence type)

# ---------------------------
# Output path
# ---------------------------
output_dir = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\02_Novelty_score"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "tsne.png"
)

def tsne_and_visualize(
    result_df_encoded,
    features_column,
    tag_column='tag',  # Column containing user-defined tags
    tag_mapping=None,           # Optional mapping from tag -> display name
    tag_colors=None,            # Optional mapping from display name -> color
    perplexity=50,
    n_iter=10000,
    metric='cosine',
    random_state=42,
    save_path=None,
    title_suffix=None
):
    """
    Perform t-SNE on encoded sequences and visualize, coloring by user-defined tags.
    """

    if title_suffix is None:
        title_suffix = features_column

    # ---------------------------
    # t-SNE
    # ---------------------------
    encoded_sequences = np.vstack(result_df_encoded[features_column].values)

    tsne = TSNE(
        perplexity=perplexity,
        max_iter=n_iter,
        metric=metric,
        random_state=random_state
    )

    tsne_data = tsne.fit_transform(encoded_sequences)

    result_df_encoded['tsne_x'] = tsne_data[:, 0]
    result_df_encoded['tsne_y'] = tsne_data[:, 1]

    # ---------------------------
    # Map tags to display names
    # ---------------------------
    if tag_mapping is not None:
        result_df_encoded['category'] = result_df_encoded[tag_column].map(tag_mapping)
    else:
        result_df_encoded['category'] = result_df_encoded[tag_column]

    # ---------------------------
    # Define colors
    # ---------------------------
    if tag_colors is None:
        unique_categories = result_df_encoded['category'].unique()
        palette = sns.color_palette("tab10", n_colors=len(unique_categories))
        tag_colors = dict(zip(unique_categories, palette))

    # ---------------------------
    # Plot
    # ---------------------------
    plt.figure(figsize=(5, 5))

    sns.scatterplot(
        x='tsne_x',
        y='tsne_y',
        hue='category',
        palette=tag_colors,
        data=result_df_encoded,
        alpha=0.7,
        s=30
    )

    plt.title(f't-SNE of {title_suffix}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    # Legend inside the plot (bottom left)
    plt.legend(
        title='Sequence type',
        loc='lower left',
        frameon=True,
        framealpha=0.9,
        fontsize=10
    )

    plt.tight_layout()

    # ---------------------------
    # Save
    # ---------------------------
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600)

    plt.show()


## Apply
tag_mapping = {
    'training_data': 'Training',
    'generated_nonfinetuned': 'Non-fine-tuned',
    'generated_finetuned': 'Fine-tuned',
    'generated_reinforced': 'Reinforced'
}

tag_colors = {
    'Training': 'black',
    'Non-fine-tuned': 'blue',
    'Fine-tuned': 'orange',
    'Reinforced': 'brown'
}

tsne_and_visualize(
    plot_df,
    features_column='cls_ESM2_emb',
    tag_column='tag',
    tag_mapping=tag_mapping,
    tag_colors=tag_colors,
    save_path=output_path
)


# In[69]:


## t-SNE plotting (novelty score)

# ---------------------------
# Output path
# ---------------------------
output_dir = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\02_Novelty_score"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "tsne_B&w_topcolorbar.png")


def truncate_colormap(cmap, minval=0.15, maxval=1.0, n=256):
    """Remove the pure white from the colormap."""
    return LinearSegmentedColormap.from_list(
        "truncated_greys",
        cmap(np.linspace(minval, maxval, n))
    )


def tsne_and_visualize(
    result_df_encoded,
    features_column,
    value_column='dist_to_reference',
    perplexity=50,
    n_iter=10000,
    metric='cosine',
    random_state=42,
    save_path=None,
    title_suffix=None
):

    if title_suffix is None:
        title_suffix = features_column

    # ---------------------------
    # Prepare embeddings
    # ---------------------------
    encoded_sequences = np.vstack(result_df_encoded[features_column].values)

    # ---------------------------
    # Run t-SNE
    # ---------------------------
    tsne = TSNE(
        perplexity=perplexity,
        max_iter=n_iter,
        metric=metric,
        random_state=random_state
    )

    tsne_data = tsne.fit_transform(encoded_sequences)

    result_df_encoded['tsne_x'] = tsne_data[:, 0]
    result_df_encoded['tsne_y'] = tsne_data[:, 1]

    # ---------------------------
    # Stabilize color scale
    # ---------------------------
    vmin = result_df_encoded[value_column].quantile(0.01)
    vmax = result_df_encoded[value_column].quantile(0.99)

    # ---------------------------
    # Grayscale without white
    # ---------------------------
    base_cmap = plt.cm.Greys   # light → dark
    cmap = truncate_colormap(base_cmap, 0.15, 1.0)

    # ---------------------------
    # Plot
    # ---------------------------
    fig, ax = plt.subplots(figsize=(4.8, 6))

    scatter = ax.scatter(
        result_df_encoded['tsne_x'],
        result_df_encoded['tsne_y'],
        c=result_df_encoded[value_column],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=30,
        alpha=0.9
    )

    # ---------------------------
    # Colorbar above the plot
    # ---------------------------
    cbar = fig.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.15)
    cbar.set_label(value_column)

    ax.set_title(f't-SNE of {title_suffix}')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    plt.show()


# ---------------------------
# Apply
# ---------------------------
tsne_and_visualize(
    plot_df,
    features_column='cls_ESM2_emb',
    value_column='dist_to_reference',
    save_path=output_path
)


# In[43]:


## t-SNE plotting (by viability)

# ---------------------------
# Output path
# ---------------------------
output_dir = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\02_Novelty_score"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "tsne_viab.png")


def tsne_and_visualize(
    result_df_encoded,
    features_column,
    label_column='pred_label',
    perplexity=50,
    n_iter=10000,
    metric='cosine',
    random_state=42,
    save_path=None,
    title_suffix=None
):

    if title_suffix is None:
        title_suffix = features_column

    # ---------------------------
    # Prepare embeddings
    # ---------------------------
    encoded_sequences = np.vstack(result_df_encoded[features_column].values)

    # ---------------------------
    # Run t-SNE
    # ---------------------------
    tsne = TSNE(
        perplexity=perplexity,
        max_iter=n_iter,
        metric=metric,
        random_state=random_state
    )

    tsne_data = tsne.fit_transform(encoded_sequences)

    result_df_encoded['tsne_x'] = tsne_data[:, 0]
    result_df_encoded['tsne_y'] = tsne_data[:, 1]

    # ---------------------------
    # Label mapping
    # ---------------------------
    label_names = {
        0: 'Negative',
        1: 'Positive'
    }

    label_colors = {
        0: 'red',  # orange/red
        1: 'green'   # green
    }

    # ---------------------------
    # Plot
    # ---------------------------
    plt.figure(figsize=(5, 5))

    for label in sorted(result_df_encoded[label_column].unique()):

        subset = result_df_encoded[result_df_encoded[label_column] == label]

        plt.scatter(
            subset['tsne_x'],
            subset['tsne_y'],
            color=label_colors[label],
            label=label_names[label],
            s=30,
            alpha=0.8
        )

    plt.legend(title="Predicted label")

    plt.title(f't-SNE of {title_suffix}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600)

    plt.show()


# ---------------------------
# Apply
# ---------------------------
tsne_and_visualize(
    plot_df,
    features_column='cls_ESM2_emb',
    label_column='pred_label',
    save_path=output_path
)


# In[ ]:




