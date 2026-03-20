#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_distances

from tqdm import tqdm
import torch
import esm


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
pos_df_formatted = pos_df[['sequence_id', 'sequence']].copy()
pos_df_formatted['prob_1'] = 1.0
pos_df_formatted['pred_label'] = 1
pos_df_formatted['tag'] = 'training_positives'

# For negatives
neg_df_formatted = neg_df[['sequence_id', 'sequence']].copy()
neg_df_formatted['prob_1'] = 0.0
neg_df_formatted['pred_label'] = 0
neg_df_formatted['tag'] = 'training_negatives'

# Combine positives and negatives into one training DataFrame
train_formatted_df = pd.concat([pos_df_formatted, neg_df_formatted], ignore_index=True)

# Check final format
print(train_formatted_df.head())
print(train_formatted_df.tail())
print(train_formatted_df.shape)
print(train_formatted_df['tag'].value_counts())


# In[3]:


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
# Split the dfs, filter by prob_1, and add tag
# ---------------------------

finetuned_08_99_df = (
    dfs['finetuned_08_unique_sequences']
    .loc[dfs['finetuned_08_unique_sequences']['prob_1'] >= 0.99]
    .reset_index(drop=True)
)
finetuned_08_99_df['tag'] = 'generated_ft08'

finetuned_12_99_df = (
    dfs['finetuned_12_unique_sequences']
    .loc[dfs['finetuned_12_unique_sequences']['prob_1'] >= 0.99]
    .reset_index(drop=True)
)
finetuned_12_99_df['tag'] = 'generated_ft12'

reinforced_08_99_df = (
    dfs['reinforced_08_unique_sequences']
    .loc[dfs['reinforced_08_unique_sequences']['prob_1'] >= 0.99]
    .reset_index(drop=True)
)
reinforced_08_99_df['tag'] = 'reinforced_08'

reinforced_12_99_df = (
    dfs['reinforced_12_unique_sequences']
    .loc[dfs['reinforced_12_unique_sequences']['prob_1'] >= 0.99]
    .reset_index(drop=True)
)
reinforced_12_99_df['tag'] = 'reinforced_12'

# Check counts and tags
print(f"finetuned_08_99: {len(finetuned_08_99_df)} rows, tag={finetuned_08_99_df['tag'].unique()}")
print(f"finetuned_12_99: {len(finetuned_12_99_df)} rows, tag={finetuned_12_99_df['tag'].unique()}")
print(f"reinforced_08_99: {len(reinforced_08_99_df)} rows, tag={reinforced_08_99_df['tag'].unique()}")
print(f"reinforced_12_99: {len(reinforced_12_99_df)} rows, tag={reinforced_12_99_df['tag'].unique()}")

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

# finetuned_08_99 + reference
# Append the reference
finetuned_08_df_99 = pd.concat([finetuned_08_99_df, ref_df], ignore_index=True)

# finetuned_12_99 + reference
#Append the reference
finetuned_12_df_99 = pd.concat([finetuned_12_99_df, ref_df], ignore_index=True)

# reinforced_08_99 + reference
# Append the reference
reinforced_08_df_99 = pd.concat([reinforced_08_99_df, ref_df], ignore_index=True)

# reinforced_12_99 + reference
# Append the reference
reinforced_12_df_99 = pd.concat([reinforced_12_99_df, ref_df], ignore_index=True)


# In[6]:


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
    df.drop(columns="_tmp_id", inplace=True)

    return df

# -----------------------------
# Pass to array
# -----------------------------
def tensor_or_list_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, list):
        # list of tensors
        return np.array([t.item() if isinstance(t, torch.Tensor) else t for t in x])
    else:
        return np.array(x)


# In[7]:


## Batch Processing Function

def compute_distances_batched(
    df,
    name,
    output_root,
    seq_column="sequence",
    id_column="sequence_id",
    reference_id="reference",
    batch_size=10000,
    embedding_batch_size=16,
):
    """
    Compute cosine distance to a reference sequence using ESM embeddings,
    processing the dataframe in crash-safe batches.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing sequences.

    name : str
        Dataset name (used to create output folder and files).

    output_root : str
        Root directory where results will be stored.

    seq_column : str
        Column containing amino-acid sequences.

    id_column : str
        Column containing sequence identifiers.

    reference_id : str
        Value in id_column identifying the reference sequence.

    batch_size : int
        Number of rows processed at a time.

    embedding_batch_size : int
        Batch size passed to add_esm_embeddings_cpu().
    """

    # ---------------------------
    # Prepare output folder
    # ---------------------------
    output_folder = os.path.join(output_root, f"{name}_batched")
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n=== Processing dataset: {name} ===")

    # ---------------------------
    # Compute reference embedding ONCE
    # ---------------------------
    print("Computing reference embedding...")
    reference_row = df[df[id_column] == reference_id].copy()

    if len(reference_row) == 0:
        raise ValueError(f"Reference '{reference_id}' not found.")

    reference_row = add_esm_embeddings_cpu(
        reference_row, seq_column=seq_column, batch_size=1
    )
    reference_row["cls_ESM2_emb"] = reference_row["cls_ESM2_emb"].apply(
        tensor_or_list_to_numpy
    )

    ref_emb = np.array(reference_row["cls_ESM2_emb"].iloc[0]).reshape(1, -1)

    print("Reference embedding ready.")

    # ---------------------------
    # Iterate over batches
    # ---------------------------
    N = len(df)
    n_batches = int(np.ceil(N / batch_size))

    print(f"Total rows: {N} | Batches: {n_batches}")

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, N)

        batch_file = os.path.join(output_folder, f"part_{i:04d}.csv")

        if os.path.exists(batch_file):
            print(f"Batch {i} already exists — skipping.")
            continue

        print(f"Processing batch {i} ({start}:{end})")

        batch_df = df.iloc[start:end].copy()

        # Generate embeddings for this batch only
        batch_df = add_esm_embeddings_cpu(
            batch_df, seq_column=seq_column, batch_size=embedding_batch_size
        )

        batch_df["cls_ESM2_emb"] = batch_df["cls_ESM2_emb"].apply(
            tensor_or_list_to_numpy
        )

        # Compute cosine distance
        X = np.vstack(batch_df["cls_ESM2_emb"].values)
        batch_df["dist_to_reference"] = cosine_distances(X, ref_emb).ravel()

        # Drop heavy column BEFORE saving
        batch_df.drop(columns=["cls_ESM2_emb"], inplace=True)

        batch_df.to_csv(batch_file, index=False)

        print(f"Saved {batch_file}")

    print(f"Finished batching for {name}")


# In[8]:


##Reconstruct the Final CSV (Fast + Safe)

def merge_batched_results(name, output_root):
    """
    Merge all batch CSVs into a single final CSV without loading everything into RAM.
    """

    output_folder = os.path.join(output_root, f"{name}_batched")
    part_files = sorted(glob.glob(os.path.join(output_folder, "part_*.csv")))

    if not part_files:
        raise ValueError("No batch files found.")

    final_output = os.path.join(output_folder, f"{name}_FULL.csv")

    print(f"Merging {len(part_files)} parts into {final_output}")

    with open(final_output, "w", encoding="utf-8") as fout:
        for i, file in enumerate(part_files):
            with open(file, "r", encoding="utf-8") as fin:
                if i == 0:
                    fout.write(fin.read())
                else:
                    next(fin)
                    fout.write(fin.read())

    print("Merge complete.")


# In[8]:


## finetuned_08_df_99

output_root = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\02_Novelty_score"

compute_distances_batched(finetuned_08_df_99, "finetuned_08_99", output_root)
merge_batched_results("finetuned_08_99", output_root)


# In[8]:


## finetuned_12_df_99

output_root = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\02_Novelty_score"

compute_distances_batched(finetuned_12_df_99, "finetuned_12_99", output_root)
merge_batched_results("finetuned_12_99", output_root)


# In[ ]:


## reinforced_12_df_99

output_root = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\02_Novelty_score"

compute_distances_batched(reinforced_12_df_99, "reinforced_12_99", output_root)
merge_batched_results("reinforced_12_99", output_root)


# In[4]:


# merge both CSVs

input_root08 = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\02_Novelty_score\finetuned_08_99_batched"
input_root12 = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\02_Novelty_score\finetuned_12_99_batched"

# File paths (adjust names if needed)
file1 = os.path.join(input_root08, "finetuned_08_99_FULL.csv")
file2 = os.path.join(input_root12, "finetuned_12_99_FULL.csv")

# Load both CSVs
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Concatenate (stack vertically)
combined_df = pd.concat([df1, df2], axis=0, ignore_index=True)

# Save to new CSV
output_root = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\02_Novelty_score"
output_file = os.path.join(output_root, "finetuned_08_12_99_FULL.csv")
combined_df.to_csv(output_file, index=False)

print("Combined CSV saved to:", output_file)


# In[ ]:




