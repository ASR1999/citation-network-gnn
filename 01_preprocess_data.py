import ijson
import pandas as pd
from tqdm import tqdm
import os
import sys

# --- Configuration ---
JSON_FILE = 'data/dblp-v12.json'
NODES_FILE = 'data/nodes.csv'
EDGES_FILE = 'data/edges.csv'
# ---------------------

def preprocess_json():
    # Make data directory if it doesn't exist
    os.makedirs(os.path.dirname(JSON_FILE), exist_ok=True)
    
    if not os.path.exists(JSON_FILE):
        print(f"Error: {JSON_FILE} not found.")
        print("Please run 'python data/download_dataset.py' first.")
        sys.exit()
    
    print(f"Starting preprocessing of {JSON_FILE}...")
    nodes_data = []
    edges_data = []
    
    # Use ijson to iteratively parse the massive JSON file
    with open(JSON_FILE, 'rb') as f:
        parser = ijson.items(f, 'item')
        for paper in tqdm(parser, desc="Parsing Papers"):
            try:
                # We only require an id and a title.
                # We will keep nodes with missing years (default to 0)
                # and filter them out during the training split.
                if 'id' not in paper or 'title' not in paper:
                    continue
                
                # --- FIX: Ensure IDs are treated as strings ---
                paper_id = str(paper['id'])
                # Default to 0 if year is missing or invalid
                year = int(paper.get('year', 0))
                title = paper.get('title', '')
                # Default to empty string if abstract is missing
                abstract = paper.get('abstract', '')
                
                # Add to nodes list
                nodes_data.append({
                    'paper_id': paper_id,
                    'year': year,
                    'title': title,
                    'abstract': abstract
                })
                
                # Add to edges list
                if 'references' in paper:
                    for ref_id in paper['references']:
                        edges_data.append({
                            'source_id': paper_id,
                            # --- FIX: Ensure target IDs are also strings ---
                            'target_id': str(ref_id)
                        })
            except Exception as e:
                print(f"Error processing a paper: {e}. Skipping.")
    
    # Create DataFrames and save
    print("Saving nodes to CSV...")
    nodes_df = pd.DataFrame(nodes_data)
    # Ensure column types are correct before saving
    nodes_df['paper_id'] = nodes_df['paper_id'].astype(str)
    nodes_df['year'] = nodes_df['year'].astype(int)
    nodes_df.to_csv(NODES_FILE, index=False)
    
    print("Saving edges to CSV...")
    edges_df = pd.DataFrame(edges_data)
    # Ensure column types are correct before saving
    edges_df['source_id'] = edges_df['source_id'].astype(str)
    edges_df['target_id'] = edges_df['target_id'].astype(str)
    edges_df.to_csv(EDGES_FILE, index=False)
    
    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess_json()

