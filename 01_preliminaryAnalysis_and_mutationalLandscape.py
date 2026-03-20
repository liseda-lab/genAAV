#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[1]:


import pandas as pd
import numpy as np
import glob
import os

from Bio import SeqIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.Seq import Seq

from sklearn.metrics.pairwise import cosine_distances

from tqdm import tqdm
import torch
import esm
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns


# ### Load sequences

# In[2]:


# ---------------------------
# Generated sequences
# ---------------------------
dfs = {}

folder_path = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Generated_sequences"
txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

for file in txt_files:
    file_name = os.path.splitext(os.path.basename(file))[0]

    with open(file, "r") as f:
        sequences = [line.strip() for line in f if line.strip()]

    df = pd.DataFrame({
        "sequence_id": [f"{file_name}_{i+1}" for i in range(len(sequences))],
        "sequence": sequences
    })

    dfs[file_name] = df
    print(f"Loaded: {file_name} ({df.shape[0]} rows, {df.shape[1]} cols)")


# In[4]:


# ---------------------------
# Load original reference dataset
# ---------------------------
file_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\01_Data_gathering_and_processing\data_2.csv'
data_2_df = pd.read_csv(file_path)

original_sequences = set(
    data_2_df['Sequence']
    .dropna()
    .astype(str)
    .unique()
)


# ### Check uniqueness 

# In[ ]:


## Check unique sequences within each set

# ---------------------------
# Summary over all DataFrames in dfs
# ---------------------------
summary_list = []

for name, df in dfs.items():
    if 'sequence' not in df.columns:
        continue  # safety check

    total_seqs = len(df)
    unique_seqs = df['sequence'].nunique()

    unique_seq_values = df['sequence'].astype(str).unique()
    match_count = sum(seq in original_sequences for seq in unique_seq_values)

    summary_list.append({
        "DataFrame": name,
        "Total Sequences": total_seqs,
        "Unique Sequences": unique_seqs,
        "Percent Unique": round(unique_seqs / total_seqs * 100, 2),
        "Unique Matches with Reference Set": match_count,
        "Percent Unique Matches": round(match_count / unique_seqs * 100, 2)
    })

# ---------------------------
# Create summary DataFrame
# ---------------------------
summary_df = pd.DataFrame(summary_list)


# In[6]:


## Check interseaction  across sets

# ---------------------------
# Load training dataset
# ---------------------------
csv_path = r'C:\Users\rodri\OneDrive\Desktop\embCOMP\01_Data_gathering_and_processing\data_2.csv'
csv_df = pd.read_csv(csv_path)

# Ensure column name consistency
csv_df = csv_df.rename(columns={'Sequence': 'sequence'})

dfs['training_data'] = csv_df
print(f"Loaded: data_2 ({csv_df.shape[0]} rows, {csv_df.shape[1]} cols)")

# ---------------------------
# Compute 6x6 matrix of unique matching sequences
# ---------------------------
file_names = list(dfs.keys())
matrix = pd.DataFrame(0, index=file_names, columns=file_names)

# Convert sequences to sets (unique, no double counting)
unique_seqs = {
    name: set(dfs[name]['sequence'].astype(str))
    for name in file_names
}

for f1 in file_names:
    for f2 in file_names:
        matrix.loc[f1, f2] = len(unique_seqs[f1].intersection(unique_seqs[f2]))

# ---------------------------
# Display result
# ---------------------------
print("\nNumber of unique matching sequences between each pair of DataFrames:")
matrix


# In[ ]:


## Get unique sequences per strategy and across strategies

strategies = [
    'finetuned_08', 'finetuned_12',
    'nonfinetuned_08', 'nonfinetuned_12',
    'reinforced_08', 'reinforced_12'
]

all_seqs = []

for strategy in strategies:
    df = dfs[strategy][['sequence_id', 'sequence']].copy()

    # --- Make representation identical to what overlap matrix used ---
    df['sequence'] = (
        df['sequence']
        .astype(str)
        .str.strip()      # remove hidden whitespace/newlines
        .str.upper()      # optional, but strongly recommended
    )

    # --- IMPORTANT: remove intra-strategy duplicates ---
    df = df.drop_duplicates(subset='sequence')

    df['strategy'] = strategy
    all_seqs.append(df)

combined_df = pd.concat(all_seqs, ignore_index=True)

# Count in how many strategies each sequence appears
seq_counts = combined_df['sequence'].value_counts()

# Keep sequences that appear in exactly ONE strategy
unique_across_all = seq_counts[seq_counts == 1].index

unique_df = combined_df[combined_df['sequence'].isin(unique_across_all)].copy()


# In[8]:


# Split back into 6 DataFrames, one per strategy
finetuned_08_df = unique_df[unique_df['strategy'] == 'finetuned_08'][['sequence_id', 'sequence']].reset_index(drop=True)
finetuned_12_df = unique_df[unique_df['strategy'] == 'finetuned_12'][['sequence_id', 'sequence']].reset_index(drop=True)
nonfinetuned_08_df = unique_df[unique_df['strategy'] == 'nonfinetuned_08'][['sequence_id', 'sequence']].reset_index(drop=True)
nonfinetuned_12_df = unique_df[unique_df['strategy'] == 'nonfinetuned_12'][['sequence_id', 'sequence']].reset_index(drop=True)
reinforced_08_df = unique_df[unique_df['strategy'] == 'reinforced_08'][['sequence_id', 'sequence']].reset_index(drop=True)
reinforced_12_df = unique_df[unique_df['strategy'] == 'reinforced_12'][['sequence_id', 'sequence']].reset_index(drop=True)

# Print counts
print("finetuned_08:", len(finetuned_08_df))
print("finetuned_12:", len(finetuned_12_df))
print("nonfinetuned_08:", len(nonfinetuned_08_df))
print("nonfinetuned_12:", len(nonfinetuned_12_df))
print("reinforced_08:", len(reinforced_08_df))
print("reinforced_12:", len(reinforced_12_df))


# In[9]:


# Get viability labels - Fine-tuned 0.8

# Files path
file_path = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\01_Preliminary sequences analysis\Viability_classifier_output"

# Fine-tuned 0.8 CLS
file_name = 'ESM2_CLS_predictions_finetuned_08_sequences.tsv'
full_path = os.path.join(file_path, file_name)
fine_tuned_08_viability_df = pd.read_csv(full_path, sep='\t')

# Get percentage positives
unique_df = fine_tuned_08_viability_df.drop_duplicates(subset='raw_sequence')
percentage_ones = unique_df['pred_label'].mean() * 100
print(f"{percentage_ones:.2f}%")

# Confirm interseaction
id_intersection = set(finetuned_08_df['sequence']) & set(fine_tuned_08_viability_df['raw_sequence'])

print("Number of matching sequences:", len(id_intersection))

## Add viability by lookup
viability_lookup = (
    fine_tuned_08_viability_df
    .drop_duplicates(subset='raw_sequence', keep='first')
    .set_index('raw_sequence')
)


finetuned_08_df['prob_1'] = finetuned_08_df['sequence'].map(
    viability_lookup['prob_1']
)

finetuned_08_df['pred_label'] = finetuned_08_df['sequence'].map(
    viability_lookup['pred_label']
)

# Inspect
finetuned_08_df


# In[10]:


# Get viability labels - Fine-tuned 1.2

# Files path
file_path = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\01_Preliminary sequences analysis\Viability_classifier_output"

# Fine-tuned 1.2 CLS
file_name = 'ESM2_CLS_predictions_finetuned_12_sequences.tsv'
full_path = os.path.join(file_path, file_name)
fine_tuned_12_viability_df = pd.read_csv(full_path, sep='\t')

# Get percentage positives
unique_df = fine_tuned_12_viability_df.drop_duplicates(subset='raw_sequence')
percentage_ones = unique_df['pred_label'].mean() * 100
print(f"{percentage_ones:.2f}%")

# Confirm interseaction
id_intersection = set(finetuned_12_df['sequence']) & set(fine_tuned_12_viability_df['raw_sequence'])

print("Number of matching sequences:", len(id_intersection))

## Add viability by lookup
viability_lookup = (
    fine_tuned_12_viability_df
    .drop_duplicates(subset='raw_sequence', keep='first')
    .set_index('raw_sequence')
)


finetuned_12_df['prob_1'] = finetuned_12_df['sequence'].map(
    viability_lookup['prob_1']
)

finetuned_12_df['pred_label'] = finetuned_12_df['sequence'].map(
    viability_lookup['pred_label']
)

# Inspect
finetuned_12_df


# In[11]:


# Get viability labels - Nonfine-tuned 0.8

# Files path
file_path = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\01_Preliminary sequences analysis\Viability_classifier_output"

# Nonfine-tuned 0.8 CLS
file_name = 'ESM2_CLS_predictions_nonfinetuned_08_sequences.tsv'
full_path = os.path.join(file_path, file_name)
nonfine_tuned_08_viability_df = pd.read_csv(full_path, sep='\t')

# Get percentage positives
unique_df = nonfine_tuned_08_viability_df.drop_duplicates(subset='raw_sequence')
percentage_ones = unique_df['pred_label'].mean() * 100
print(f"{percentage_ones:.2f}%")

# Confirm interseaction
id_intersection = set(nonfinetuned_08_df['sequence']) & set(nonfine_tuned_08_viability_df['raw_sequence'])

print("Number of matching sequences:", len(id_intersection))

## Add viability by lookup
viability_lookup = (
    nonfine_tuned_08_viability_df
    .drop_duplicates(subset='raw_sequence', keep='first')
    .set_index('raw_sequence')
)


nonfinetuned_08_df['prob_1'] = nonfinetuned_08_df['sequence'].map(
    viability_lookup['prob_1']
)

nonfinetuned_08_df['pred_label'] = nonfinetuned_08_df['sequence'].map(
    viability_lookup['pred_label']
)

# Inspect
nonfinetuned_08_df


# In[12]:


# Get viability labels - Nonfine-tuned 1.2

# Files path
file_path = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\01_Preliminary sequences analysis\Viability_classifier_output"

# Nonfine-tuned 1.2 CLS
file_name = 'ESM2_CLS_predictions_nonfinetuned_12_sequences.tsv'
full_path = os.path.join(file_path, file_name)
nonfine_tuned_12_viability_df = pd.read_csv(full_path, sep='\t')

# Get percentage positives
unique_df = nonfine_tuned_12_viability_df.drop_duplicates(subset='raw_sequence')
percentage_ones = unique_df['pred_label'].mean() * 100
print(f"{percentage_ones:.2f}%")

# Confirm intersection
id_intersection = set(nonfinetuned_12_df['sequence']) & set(nonfine_tuned_12_viability_df['raw_sequence'])

print("Number of matching sequences:", len(id_intersection))

## Add viability by lookup
viability_lookup = (
    nonfine_tuned_12_viability_df
    .drop_duplicates(subset='raw_sequence', keep='first')
    .set_index('raw_sequence')
)


nonfinetuned_12_df['prob_1'] = nonfinetuned_12_df['sequence'].map(
    viability_lookup['prob_1']
)

nonfinetuned_12_df['pred_label'] = nonfinetuned_12_df['sequence'].map(
    viability_lookup['pred_label']
)

# Inspect
nonfinetuned_12_df


# In[13]:


# Get viability labels - Reinforced 0.8

# Files path
file_path = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\01_Preliminary sequences analysis\Viability_classifier_output"

# Reinforced 0.8 CLS
file_name = 'ESM2_CLS_predictions_reinforced_08_sequences.tsv'
full_path = os.path.join(file_path, file_name)
reinforced_08_viability_df = pd.read_csv(full_path, sep='\t')

# Get percentage positives
unique_df = reinforced_08_viability_df.drop_duplicates(subset='raw_sequence')
percentage_ones = unique_df['pred_label'].mean() * 100
print(f"{percentage_ones:.2f}%")

# Confirm interseaction
id_intersection = set(reinforced_08_df['sequence']) & set(reinforced_08_viability_df['raw_sequence'])

print("Number of matching sequences:", len(id_intersection))

## Add viability by lookup
viability_lookup = (
    reinforced_08_viability_df
    .drop_duplicates(subset='raw_sequence', keep='first')
    .set_index('raw_sequence')
)


reinforced_08_df['prob_1'] = reinforced_08_df['sequence'].map(
    viability_lookup['prob_1']
)

reinforced_08_df['pred_label'] = reinforced_08_df['sequence'].map(
    viability_lookup['pred_label']
)

# Inspect
reinforced_08_df


# In[14]:


# Get viability labels - Reinforced 1.2

# Files path
file_path = r"C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\01_Preliminary sequences analysis\Viability_classifier_output"

# Reinforced 1.2 CLS
file_name = 'ESM2_CLS_predictions_reinforced_12_sequences.tsv'
full_path = os.path.join(file_path, file_name)
reinforced_12_viability_df = pd.read_csv(full_path, sep='\t')

# Get percentage positives
unique_df = reinforced_12_viability_df.drop_duplicates(subset='raw_sequence')
percentage_ones = unique_df['pred_label'].mean() * 100
print(f"{percentage_ones:.2f}%")

# Confirm interseaction
id_intersection = set(reinforced_12_df['sequence']) & set(reinforced_12_viability_df['raw_sequence'])

print("Number of matching sequences:", len(id_intersection))

## Add viability by lookup
viability_lookup = (
    reinforced_12_viability_df
    .drop_duplicates(subset='raw_sequence', keep='first')
    .set_index('raw_sequence')
)


reinforced_12_df['prob_1'] = reinforced_12_df['sequence'].map(
    viability_lookup['prob_1']
)

reinforced_12_df['pred_label'] = reinforced_12_df['sequence'].map(
    viability_lookup['pred_label']
)

# Inspect
reinforced_12_df


# In[15]:


## Export to CSV
# Define output folder (change path if needed)
output_folder =  r'C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\01_Preliminary sequences analysis'

# Export each DataFrame
finetuned_08_df.to_csv(f"{output_folder}/finetuned_08_unique_sequences.csv", index=False)
finetuned_12_df.to_csv(f"{output_folder}/finetuned_12_unique_sequences.csv", index=False)
nonfinetuned_08_df.to_csv(f"{output_folder}/nonfinetuned_08_unique_sequences.csv", index=False)
nonfinetuned_12_df.to_csv(f"{output_folder}/nonfinetuned_12_unique_sequences.csv", index=False)
reinforced_08_df.to_csv(f"{output_folder}/reinforced_08_unique_sequences.csv", index=False)
reinforced_12_df.to_csv(f"{output_folder}/reinforced_12_unique_sequences.csv", index=False)

print("All CSVs exported successfully!")


# ### Evaluate size distribution

# In[16]:


## Function to plot size distribution
def plot_sequence_length_histogram(
    df,
    bins=30,
    seq_col='sequence',
    reference_length=735,
    title="Sequence Length Distribution",
    save_path=None
):
    """
    Plot a histogram of sequence length distribution with a reference marker.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing sequences
        bins (int): Number of histogram bins
        seq_col (str): Column containing sequences
        reference_length (int): Reference sequence length to mark
        title (str): Plot title
        save_path (str): Path to save the figure (if None, figure is not saved)
    """
    # Compute sequence lengths
    seq_lengths = df[seq_col].astype(str).str.len()

    plt.figure(figsize=(10,6))
    plt.hist(seq_lengths, bins=bins, color='skyblue', edgecolor='black')
    
    # Reference length marker
    plt.axvline(
        reference_length,
        linestyle="--",
        linewidth=2,
        color='red',
        label=f"Reference length = {reference_length}"
    )

    plt.xlabel("Sequence length (aa)")
    plt.ylabel("Count")
    plt.title(title)

    # Legend outside plot
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# In[17]:


## Visualize size distribution for fine-tuned sequences 0.8
plot_sequence_length_histogram(
    finetuned_08_df,
    bins=15,
    title="Length distribution of fine-tuned sequences 0.8 ",
    save_path="finetuned_08_length_hist.png"
)


# In[18]:


## Visualize size distribution for fine-tuned sequences 1.2
plot_sequence_length_histogram(
    finetuned_12_df,
    bins=15,
    title="Length distribution of fine-tuned sequences 1.2 ",
    save_path="finetuned_1.2_length_hist.png"
)


# In[19]:


## Visualize size distribution for non-fine-tuned sequences 0.8
plot_sequence_length_histogram(
    nonfinetuned_08_df,
    bins=15,
    title="Length distribution of non-fine-tuned sequences 0.8 ",
    save_path="non_finetuned_08_length_hist.png"
)


# In[20]:


## Visualize size distribution for non-fine-tuned sequences 1.2
plot_sequence_length_histogram(
    nonfinetuned_12_df,
    bins=15,
    title="Length distribution of non-fine-tuned sequences 1.2 ",
    save_path="non_finetuned_1.2_length_hist.png"
)


# ### Mutation landscape

# In[21]:


## Function detect changes
def detect_changes(s):
    """
    Analyzes the result of a pairwise sequence alignment and returns a list of differences 
    between the mutated and original sequences.

    Parameters:
    s (str): A string containing the output of a pairwise alignment in the form of three lines:
             - mutated sequence
             - alignment matches (using '|' for matches, ' ' for mismatches/gaps)
             - original sequence
             followed by two unused lines (usually empty).

    Returns:
    List[List]: A list of changes detected in the following formats:
                - ["Ins", aa, pos]             → Insertion of amino acid 'aa' at position 'pos'
                - ["Del", aa, pos]             → Deletion of amino acid 'aa' from position 'pos'
                - ["Sub", old_aa, new_aa, pos] → Substitution of 'old_aa' with 'new_aa' at position 'pos'

    Notes:
    - The function merges adjacent insertions and deletions into substitutions when possible.
    - Positions are based on the index in the original sequence (ignoring gaps).
    """

    # The algorithm considers 3 strings: mutated, matches, and original (the 3 lines from the alignment)
    mutated, matches, original, _, _ = s.split("\n")
    
    # This algorithm switches "mode" under certain conditions
    mode = None
    
    # The algorithm needs to remember previous insertions/deletions in order to merge them into substitutions
    # It uses two stacks to "remember" what it read previously
    insert_stack = []
    delete_stack = []
    
    # List to store the final results
    results = []
    
    # Counter for the "real" position in the original string (ignoring gaps)
    c = 0
    
    # For each position in the alignment match line
    for i, m in enumerate(matches):

        # If there is a match at this position (i.e., no change)
        if m == "|":
            
            # Record any previously read deletions
            for elem in delete_stack:
                aa, pos = elem
                results.append(["Del", aa, pos])
                
            # Record any previously read insertions
            for elem in insert_stack:
                aa, pos = elem
                results.append(["Ins", aa, pos])
                
            # Clear the stacks as they have now been processed
            delete_stack = []
            insert_stack = []
            
            # Reset mode to Inactive
            mode = None
        
        # Determine the current mode based on what's at position i
        
        # If the original has an amino acid and mutated has a gap → deletion
        if original[i] != "-" and mutated[i] == "-":
            mode = "Del"
        
        # If the original has a gap and mutated has an amino acid → insertion
        if original[i] == "-" and mutated[i] != "-":
            mode = "Ins"

        # Perform actions based on the current mode
        if mode == "Del":
            if len(insert_stack) > 0:
                # Merge with previous insertion → substitution
                aa, index = insert_stack.pop()
                results.append(["Sub", original[i], aa, c])
            else:
                # Record deletion
                delete_stack.append([original[i], c])
        
        elif mode == "Ins":
            if len(delete_stack) > 0:
                # Merge with previous deletion → substitution
                aa, index = delete_stack.pop()
                results.append(["Sub", aa, mutated[i], c])
            else:
                # Record insertion
                insert_stack.append([mutated[i], c])
        
        # Advance the real position counter if not reading a gap in the original
        if original[i] != "-":
            c += 1
            
    # Final cleanup: write any remaining insertions or deletions to results
    for elem in delete_stack:
        aa, pos = elem
        results.append(["Del", aa, pos])
    for elem in insert_stack:
        aa, pos = elem
        results.append(["Ins", aa, pos - 1])
    
    # Reset (good practice)
    delete_stack = []
    insert_stack = []
    mode = None
    
    # Return sorted list of operations by position
    return sorted(results, key=lambda x: x[-1])


# ### Create 'changes matrix' for each df

# In[22]:


# AAV2 VP1 reference reference
aav2vp1_refSeq = "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"

## Function to crate the changes matrix
def make_changes_matrix(df, refSeq):
    # Initialize an empty NumPy array with 4 rows (Unchanged, Sub, Ins, Del) and N columns (length of the reference sequence)
    array = np.zeros((4, len(refSeq)), dtype=int)
    
    # Loop through each sequence in the df
    for index, row in df.iterrows():
        # Access the 'Sequence_Label' and 'Aligned_Piece' columns
        tag = row['sequence_id']
        sequence = row['sequence']
        
        # Perform pairwise alignment with the reference sequence
        alignments = pairwise2.align.globalxx(sequence, refSeq, one_alignment_only=True)
        
        # Format and get the changes for the current sequence
        s = format_alignment(*alignments[0])
        changes = detect_changes(s)
        
        # Initialize a matrix for the current sequence
        matrix = np.zeros((4, len(refSeq)), dtype=int)
        
        # Define unchanged positions
        unchanged_positions = set(range(len(refSeq)))
        
        # Populate the matrix based on the changes
        for change_type, *rest in changes:
            if change_type == "Ins":
                matrix[2][rest[1]] = 1
                unchanged_positions.discard(rest[1])
            elif change_type == "Del":
                matrix[3][rest[1]] = 1
                unchanged_positions.discard(rest[1])
            elif change_type == "Sub":
                matrix[1][rest[2]] = 1
                unchanged_positions.discard(rest[2])
        
        # Populate the Unchanged row
        matrix[0, list(unchanged_positions)] = 1
        
        # Add the current matrix to the result_array
        array += matrix
    
    return array


# In[23]:


## Application to nonfinetuned (all)

## Merge both temps 
nonfinetuned = pd.concat([nonfinetuned_08_df, nonfinetuned_12_df], ignore_index=True)

## Application of 'make_changes matrix function'
nonfinetuned_df = make_changes_matrix(nonfinetuned, refSeq=aav2vp1_refSeq)

# Save the df to a CSV file
df_result = pd.DataFrame(nonfinetuned_df)
save_path = r'C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\01_Preliminary sequences analysis\ChangesMatrix_nonfinetuned_08_12.csv'
df_result.to_csv(save_path, index=False)

# Print the columns including the 561-588 (fragment) region
print('Result matrix (columns 557 to 590):')
print(df_result.iloc[:, 558:590])  
print('')


# In[24]:


## Application to finetuned (all)

## Merge both temps 
finetuned = pd.concat([finetuned_08_df, finetuned_12_df], ignore_index=True)

## Application of 'make_changes matrix function'
finetuned_df = make_changes_matrix(finetuned, refSeq=aav2vp1_refSeq)

# Save the df to a CSV file
df_result = pd.DataFrame(finetuned_df)
save_path = r'C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\01_Preliminary sequences analysis\ChangesMatrix_finetuned_08_12.csv'
df_result.to_csv(save_path, index=False)

# Print the columns including the 561-588 (fragment) region
print('Result matrix (columns 557 to 590):')
print(df_result.iloc[:, 558:590])  
print('')


# In[25]:


## Application to reinforced (all)

## Merge both temps 
reinforced = pd.concat([reinforced_08_df, reinforced_12_df], ignore_index=True)

## Application of 'make_changes matrix function'
reinforced_df = make_changes_matrix(reinforced, refSeq=aav2vp1_refSeq)

# Save the df to a CSV file
df_result = pd.DataFrame(reinforced_df)
save_path = r'C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\01_Preliminary sequences analysis\ChangesMatrix_reinforced_08_12.csv'
df_result.to_csv(save_path, index=False)

# Print the columns including the 561-588 (fragment) region
print('Result matrix (columns 557 to 590):')
print(df_result.iloc[:, 558:590])  
print('')


# ### Auxiliary functions for plotting mutation landscape

# In[26]:


### Function to plot mutation type across positions
def plot_mutation_type_distribution(df, suffix):
    # Calculate percentages for each mutation type at each position
    total_mutations = df.iloc[-1]  # Total mutations per column
    mutation_types = df.iloc[:-1] / total_mutations  # Divide each mutation type count by total mutations
    mutation_types = mutation_types.transpose() * 100  # Convert to percentages and transpose for plotting

    # Create a figure with custom size
    fig, ax = plt.subplots(figsize=(6, 5))  # Adjust the figsize parameter if needed

    # Define custom colors for stacks
    colors = ['blue', 'green', 'red', 'white']

    # Reverse the order of mutation types for stacking
    mutation_types = mutation_types.iloc[:, ::-1]

    # Plot theme river with custom colors and reversed stacking order
    stacks = ax.stackplot(range(len(mutation_types)), mutation_types.values.T, labels=mutation_types.columns, colors=colors)
    ax.set_xlabel('Amino acid position')
    ax.set_ylabel('Percentage of change')
    ax.set_title(f'{suffix}')

    # Set custom labels for x-axis every 5 positions (for zooming-in)
    positions = list(range(0, len(mutation_types), 5))
    ax.set_xticks(positions)
    ax.set_xticklabels(positions)

    # Add custom legend outside the plot area
    #legend_labels = ['Deletion', 'Insertion', 'Substitution', 'Unchanged']  # Match stack order
    #ax.legend(stacks, legend_labels, loc='upper left')

    # Limit the x-axis between 555 and 595 (for zooming-in)
    ax.set_xlim(555, 595)

    # Save the figure
    output_file = os.path.join(save_path, f"Mutation landscape {suffix}.png")
    plt.savefig(output_file, format='png', dpi=600, bbox_inches='tight')
    plt.show()


# In[27]:


## ## Application of 'plot_mutation_type_distribution' function to nonfinetuned

# Load the CSV file into a df
file_path = r'C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\01_Preliminary sequences analysis\ChangesMatrix_nonfinetuned_08_12.csv'
nonfinetuned_chMatrix = pd.read_csv(file_path)

# Add a fifth row with the total sum per column
nonfinetuned_chMatrix.loc['Total'] = nonfinetuned_chMatrix.sum()
    
#Saving path
save_path =  r'C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\01_Preliminary sequences analysis'

# Application of 'plot_mutation_type_distribution' function to non-finetuned sequences set
plot_mutation_type_distribution(nonfinetuned_chMatrix, 'Non-fine-tuned sequences')


# In[28]:


## ## Application of 'plot_mutation_type_distribution' function to finetuned

# Load the CSV file into a df
file_path = r'C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\01_Preliminary sequences analysis\ChangesMatrix_finetuned_08_12.csv'
finetuned_chMatrix = pd.read_csv(file_path)

# Add a fifth row with the total sum per column
finetuned_chMatrix.loc['Total'] = finetuned_chMatrix.sum()
    
#Saving path
save_path =  r'C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\01_Preliminary sequences analysis'

# Application of 'plot_mutation_type_distribution' function to finetuned sequences
plot_mutation_type_distribution(finetuned_chMatrix, 'Fine-tuned sequences')


# In[29]:


## ## Application of 'plot_mutation_type_distribution' function to reinforced

# Load the CSV file into a df
file_path = r'C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\01_Preliminary sequences analysis\ChangesMatrix_reinforced_08_12.csv'
reinforced_chMatrix = pd.read_csv(file_path)

# Add a fifth row with the total sum per column
reinforced_chMatrix.loc['Total'] = reinforced_chMatrix.sum()
    
#Saving path
save_path =  r'C:\Users\rodri\OneDrive\Desktop\GenAAV paper\Scripts\01_Preliminary sequences analysis'

# Application of 'plot_mutation_type_distribution' function to Viable sequences set
plot_mutation_type_distribution(reinforced_chMatrix, 'Reinforced sequences')

