import json
from typing import Dict, Any
from collections import defaultdict, Counter

class ConceptAnalyzer:
    def __init__(self, filepath: str):
        """Loads the JSON data from the specified filepath."""
        self.filepath = filepath
        self.data = self._load_data()
        
    def _load_data(self) -> Dict[str, Any]:
        """Reads the JSON file and returns it as a dictionary."""
        try:
            with open(self.filepath, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: The file {self.filepath} was not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: The file {self.filepath} contains invalid JSON.")
            return {}
        
    def print_high_entropy_concepts(self, threshold: float, min_mean_acts_sum: float = 0.0):
        """
        Prints the concept indices and the total count of concepts 
        in each class that have an entropy strictly above the threshold K
        and a sum of mean_acts strictly above min_mean_acts_sum.
        """
        if not self.data:
            print("No data loaded. Cannot perform analysis.")
            return

        concept_data = self.data.get("thresholded_concept_entropies", {})
        
        if not concept_data:
            print("No 'thresholded_concept_entropies' found in the data.")
            return

        print(f"--- Concepts with Entropy > {threshold} & Sum(mean_acts) > {min_mean_acts_sum} ---")
        
        for class_id, concepts in concept_data.items():
            # Filter by both entropy threshold AND the sum of mean_acts
            valid_concepts = [
                c for c in concepts 
                if c.get("entropy", 0) > threshold and sum(c.get("mean_acts", [])) > min_mean_acts_sum
            ]
            
            concept_idxs = [c.get("concept_index") for c in valid_concepts]
            
            print(f"Class ID: {class_id}")
            print(f"  Total Concepts Above Thresholds: {len(concept_idxs)}")
            print(f"  Concept Indices: {concept_idxs}\n")

    def analyze_concept_occurrences(self, nb_concepts: int = 7680, min_strength: float = 0.0001):
        """
        Analyzes and prints how many classes each concept appears in,
        filtered by a minimum activation strength.
        """
        if not self.data:
            print("No data loaded. Cannot perform analysis.")
            return

        concept_data = self.data.get("thresholded_concept_entropies", {})
        if not concept_data:
            print("No 'thresholded_concept_entropies' found in the data.")
            return

        # Key = concept_index, Value = number of classes it appeared in
        concept_counts = defaultdict(int)

        # Iterate over all available class keys dynamically
        for class_key, items in concept_data.items():
            concepts_in_this_class = set()
            
            for item in items:
                idx = item.get('concept_index')
                # Safely get mean_acts; default to empty list if missing
                mean_acts = item.get('mean_acts', [])
                strength = sum(mean_acts)

                if idx is not None and strength > min_strength:
                    concepts_in_this_class.add(idx)
            
            # Update the global counter
            for idx in concepts_in_this_class:
                concept_counts[idx] += 1

        # Print detailed list (Concept vs Count)
        print(f"{'Concept Index':<15} | {'Class Count'}")
        print("-" * 30)
        
        found_concepts = sorted(concept_counts.keys())
        for idx in found_concepts:
            if 0 <= idx <= nb_concepts:
                print(f"{idx:<15} | {concept_counts[idx]}")

        # Calculate and Print Summary Statistics
        usage_frequency = Counter(concept_counts.values())

        print("\n" + "="*35)
        print("SUMMARY: FREQUENCY OF OCCURRENCES")
        print("="*35)
        
        # Iterate from 1 up to the total number of classes found in the JSON
        total_classes = len(concept_data)
        for i in range(1, total_classes + 1):
            count = usage_frequency.get(i, 0)
            print(f"Occurred exactly {i} time(s): {count} concept(s)")