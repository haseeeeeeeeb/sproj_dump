

import pandas as pd
import re

def select_best_checkpoint(path, env_indices=None):
    """
    Selects the best checkpoint based on balanced out accuracies (mean - variance)
    across specified environments. 
    Automatically handles changing headers in DomainBed logs by resetting
    whenever a line with alphabetic tokens appears.
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    header = None
    data_rows = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        # Check if this line looks like a header (has any alphabets)
        if any(re.search(r'[A-Za-z]', t) for t in tokens):
            header = tokens
            data_rows = []  # reset table since this is the new header
        elif header is not None:
            # numeric data row (only after we have a header)
            data_rows.append(tokens)

    if header is None or not data_rows:
        raise ValueError("No valid metrics table found in file.")

    # Normalize all rows to match header length (some may have trailing spaces)
    num_cols = len(header)
    cleaned_rows = [r[:num_cols] for r in data_rows if len(r) >= num_cols]

    # Build DataFrame
    df = pd.DataFrame(cleaned_rows, columns=header)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # --- Identify environment columns ---
    all_out_cols = [c for c in df.columns if re.match(r'env\d+_out_acc', c)]
    if not all_out_cols:
        raise ValueError("No environment accuracy columns found.")

    if env_indices is None:
        env_cols = all_out_cols
    else:
        env_cols = [f"env{i}_out_acc" for i in env_indices if f"env{i}_out_acc" in df.columns]
        if not env_cols:
            raise ValueError(f"No valid env columns for indices {env_indices}")

    # --- Ensure numeric 'step' column exists ---
    step_col = next((c for c in df.columns if c.strip().lower() == 'step'), None)
    if step_col is None:
        raise ValueError("Couldn't find a 'step' column.")

    df[env_cols + [step_col]] = df[env_cols + [step_col]].apply(pd.to_numeric, errors='coerce')

    # --- Compute balance score ---
    df['mean_out'] = df[env_cols].mean(axis=1)
    df['var_out'] = df[env_cols].var(axis=1)
    df['balance_score'] = df['mean_out'] - df['var_out']

    # --- Select best checkpoint ---
    best_idx = df['balance_score'].idxmax()
    best_step = int(df.loc[best_idx, step_col])

    return best_step
