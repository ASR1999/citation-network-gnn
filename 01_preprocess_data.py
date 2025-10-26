import ijson
import pandas as pd
from tqdm import tqdm
import os
import csv

# --- Configuration ---
INPUT_JSON_FILE = 'data/dblp.v12.json'
OUTPUT_DIR = 'data'
NODES_FILE = os.path.join(OUTPUT_DIR, 'nodes.csv')
EDGES_FILE = os.path.join(OUTPUT_DIR, 'edges.csv')
# ---------------------

def preprocess_data():
    """
    Streams the massive DBLP JSON file and creates two CSVs:
    1. nodes.csv: [paper_id, year, title, abstract]
    2. edges.csv: [source_id, target_id] (representing a citation)
    """
    
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Starting preprocessing of {INPUT_JSON_FILE}...")
    print(f"Output nodes will be in {NODES_FILE}")
    print(f"Output edges will be in {EDGES_FILE}")
    
    # Use context managers to handle file opening and closing
    with open(INPUT_JSON_FILE, 'rb') as f_in, \
         open(NODES_FILE, 'w', newline='', encoding='utf-8') as f_nodes, \
         open(EDGES_FILE, 'w', newline='', encoding='utf-8') as f_edges:
        
        # Create CSV writers
        nodes_writer = csv.writer(f_nodes)
        nodes_writer.writerow(['paper_id', 'year', 'title', 'abstract'])
        
        edges_writer = csv.writer(f_edges)
        edges_writer.writerow(['source_id', 'target_id'])
        
        # Use ijson to parse the file iteratively
        parser = ijson.items(f_in, 'item')
        
        node_count = 0
        edge_count = 0
        
        # Use tqdm for a progress bar
        for paper in tqdm(parser, desc="Parsing DBLP JSON"):
            try:
                # We need at least an ID and a year for temporal splitting
                if 'id' not in paper or 'year' not in paper:
                    continue
                
                paper_id = paper['id']
                year = int(paper.get('year', 0))
                
                # Skip papers with invalid year
                if year == 0:
                    continue
                    
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                
                # Write node data
                nodes_writer.writerow([paper_id, year, title, abstract])
                node_count += 1
                
                # Write edge data
                if 'references' in paper:
                    for ref_id in paper['references']:
                        if ref_id: # Ensure ref_id is not empty
                            edges_writer.writerow([paper_id, ref_id])
                            edge_count += 1
                            
            except Exception as e:
                print(f"Warning: Skipping a paper due to error: {e}")
                
    print("\n--- Preprocessing Complete ---")
    print(f"Total nodes (papers) processed: {node_count}")
    print(f"Total edges (citations) processed: {edge_count}")
    print(f"Files saved: {NODES_FILE}, {EDGES_FILE}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"Error: Input file not found at {INPUT_JSON_FILE}")
        print("Please download 'dblp-v12.json' and place it in the 'data/' directory.")
    else:
        preprocess_data()
