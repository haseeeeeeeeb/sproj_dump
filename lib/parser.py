def get_top_k_steps(file_path: str, envs: list[int], k: int) -> list[int]:
    """
    Parses a training log file and returns the top k steps with the highest 
    average out-domain accuracy for the specified environments.
    
    Args:
        file_path: Path to the log text file.
        envs: A list of environment integers (e.g., [0, 1, 2, 3]).
        k: The number of top steps to return.
        
    Returns:
        A list of the top k steps.
    """
    # Define the column names we want to extract based on the passed envs
    target_cols = [f'env{env}_out_acc' for env in envs]
    results = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    header_indices = {}
    is_parsing_table = False
    
    for line in lines:
        parts = line.split()
        if not parts:
            continue
            
        # Identify the table header
        if 'env0_in_acc' in parts and 'step' in parts:
            is_parsing_table = True
            header_indices = {col: i for i, col in enumerate(parts)}
            continue
            
        # Parse table rows
        if is_parsing_table:
            try:
                # Try to cast the line to floats. If it fails (e.g., hitting a traceback 
                # or a new "Environment:" block), it falls into the except block.
                row_vals = [float(x) for x in parts]
                
                # Make sure the row has all the columns we expect
                if len(row_vals) < len(header_indices):
                    continue
                    
                # Extract step and calculate the average for the requested environments
                step = int(row_vals[header_indices['step']])
                accs = [row_vals[header_indices[col]] for col in target_cols]
                avg_acc = sum(accs) / len(accs)
                
                results.append((avg_acc, step))
                
            except (ValueError, KeyError):
                # Safely ignore stack traces, empty lines, and string headers
                continue
                
    # Sort results primarily by average accuracy (descending), 
    # and secondarily by step (ascending) as a tie-breaker
    results.sort(key=lambda x: (-x[0], x[1]))
    
    # Return just the top k step values
    top_k_steps = [step for acc, step in results[:k]]
    
    return top_k_steps