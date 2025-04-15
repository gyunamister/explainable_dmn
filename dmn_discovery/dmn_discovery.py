import sys
import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pm4py.objects.petri_net.obj import PetriNet
import json

class DMNDiscovery:
    """
    Class for discovering DMN decision tables from feature tables.
    """
    
    def __init__(self, 
                 guard_ds_per_place: Dict[PetriNet.Place, pd.DataFrame],
                 min_samples_leaf: int = 5,
                 max_depth: int = 5,
                 min_rule_support: float = 0.05,
                 verbose: bool = True):
        """
        Initialize the DMN discovery process.
        
        Args:
            guard_ds_per_place (Dict[PetriNet.Place, pd.DataFrame]): Dictionary mapping places to their feature tables
            min_samples_leaf (int): Minimum samples required to be at a leaf node
            max_depth (int): Maximum depth of the decision tree
            min_rule_support (float): Minimum support (fraction of samples) for a rule to be included
            verbose (bool): Whether to print verbose output
        """
        self.guard_ds_per_place = guard_ds_per_place
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_rule_support = min_rule_support
        self.verbose = verbose
        self.dmn_tables = {}
        self.feature_encoders = {}
        self.target_encoders = {}
        self.decision_trees = {}
        
    def _print_if_verbose(self, msg, end="\n"):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(msg, end=end)
    
    def discover_all_dmn_tables(self) -> Dict[PetriNet.Place, Dict]:
        """
        Discover DMN decision tables for all places in the feature tables.
        
        Returns:
            Dict[PetriNet.Place, Dict]: Dictionary mapping places to their DMN decision tables
        """
        self._print_if_verbose("Starting DMN discovery for all decision points...")
        
        for place, feature_table in self.guard_ds_per_place.items():
            self._print_if_verbose(f"Processing decision point '{place.name}'...", end=" ")
            
            # Check if there are enough samples
            if len(feature_table) < 2 * self.min_samples_leaf:
                self._print_if_verbose(f"Skipping - not enough samples ({len(feature_table)})")
                continue
                
            # Check if there's actually a decision to make (multiple target values)
            if feature_table['target'].nunique() < 2:
                self._print_if_verbose(f"Skipping - only one target value '{feature_table['target'].iloc[0]}'")
                continue
                
            # Discover DMN table for this place
            dmn_table = self.discover_dmn_table(place, feature_table)
            self.dmn_tables[place] = dmn_table
            
            self._print_if_verbose(f"Done - created table with {len(dmn_table['rules'])} rules")
            
        return self.dmn_tables
    
    def discover_dmn_table(self, place: PetriNet.Place, feature_table: pd.DataFrame) -> Dict:
        """
        Discover a DMN decision table for a single place.
        
        Args:
            place (PetriNet.Place): The place to discover a DMN table for
            feature_table (pd.DataFrame): The feature table for this place
            
        Returns:
            Dict: A DMN decision table in dictionary format
        """
        # 1. Prepare data
        X, y, feature_names = self._prepare_data(place, feature_table)
        
        # 2. Train decision tree
        tree = self._train_decision_tree(X, y, feature_names)
        self.decision_trees[place] = tree
        
        # 3. Extract rules from decision tree
        rules = self._extract_rules_from_tree(tree, feature_names, place)
        
        # 4. Create DMN table
        dmn_table = self._create_dmn_table(place, rules, feature_names)
        
        return dmn_table
    
    def _prepare_data(self, place: PetriNet.Place, feature_table: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for decision tree training.
        
        Args:
            place (PetriNet.Place): The place being processed
            feature_table (pd.DataFrame): The feature table
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: X, y, and feature names
        """
        # Separate features and target
        X_df = feature_table.drop('target', axis=1)
        y = feature_table['target']
        # get the label for each y instance
        y = y.apply(lambda x: x.label)
        
        # Get feature names
        feature_names = X_df.columns.tolist()
        
        # Handle categorical features with one-hot encoding
        self.feature_encoders[place] = {}
        transformed_dfs = []
        
        for col in feature_names:
            # Check if column is categorical
            if X_df[col].dtype == 'object' or X_df[col].nunique() < 10:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(X_df[[col]])
                encoded_df = pd.DataFrame(
                    encoded, 
                    columns=[f"{col}@{val}" for val in encoder.categories_[0]],
                    index=X_df.index
                )
                transformed_dfs.append(encoded_df)
                self.feature_encoders[place][col] = encoder
            else:
                # Keep numerical features as is
                transformed_dfs.append(X_df[[col]])
        
        # Combine all transformed features
        X_transformed = pd.concat(transformed_dfs, axis=1)
        
        # Encode target values
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        self.target_encoders[place] = target_encoder
        
        return X_transformed.values, y_encoded, X_transformed.columns.tolist()
    
    def _train_decision_tree(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> DecisionTreeClassifier:
        """
        Train a decision tree on the prepared data.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            feature_names (List[str]): Names of features
            
        Returns:
            DecisionTreeClassifier: Trained decision tree
        """
        tree = DecisionTreeClassifier(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=42
        )
        tree.fit(X, y)
        return tree
    
    def _extract_rules_from_tree(self, tree: DecisionTreeClassifier, feature_names: List[str], place: PetriNet.Place) -> List[Dict]:
        """
        Extract rules from a trained decision tree.
        
        Args:
            tree (DecisionTreeClassifier): Trained decision tree
            feature_names (List[str]): Names of features
            place (PetriNet.Place): Place being processed
            
        Returns:
            List[Dict]: List of rules extracted from the tree
        """
        tree_ = tree.tree_
        rules = []
        
        def extract_rule(node_id=0, conditions=None):
            if conditions is None:
                conditions = []
            
            # If we're at a leaf node, create a rule
            if tree_.children_left[node_id] == _tree.TREE_LEAF:
                # Get predicted class and sample count
                class_idx = np.argmax(tree_.value[node_id][0])
                n_samples = int(tree_.n_node_samples[node_id])
                
                # Calculate support
                support = n_samples / tree_.n_node_samples[0]
                
                # Skip rules with low support
                if support < self.min_rule_support:
                    return
                
                # Decode the target class
                target_class = self.target_encoders[place].inverse_transform([class_idx])[0]
                
                # Create the rule
                rule = {
                    'conditions': conditions.copy(),
                    'outcome': target_class,
                    'support': support,
                    'samples': n_samples
                }
                rules.append(rule)
                return
            
            # Process left branch (condition is satisfied)
            feature_idx = tree_.feature[node_id]
            threshold = tree_.threshold[node_id]
            feature_name = feature_names[feature_idx]
            # Extracted feature name (handle one-hot encoded features)
            if "@" in feature_name:
                orig_feature, value = feature_name.rsplit("@", 1)
                left_condition = {'feature': orig_feature, 'operator': '==', 'value': value}
                right_condition = {'feature': orig_feature, 'operator': '!=', 'value': value}
            else:
                left_condition = {'feature': feature_name, 'operator': '<=', 'value': threshold}
                right_condition = {'feature': feature_name, 'operator': '>', 'value': threshold}
            
            # Left branch - condition is true
            conditions.append(left_condition)
            extract_rule(tree_.children_left[node_id], conditions)
            conditions.pop()
            
            # Right branch - condition is false
            conditions.append(right_condition)
            extract_rule(tree_.children_right[node_id], conditions)
            conditions.pop()
        
        # Start recursive extraction
        extract_rule()
        return rules
    
    def _create_dmn_table(self, place: PetriNet.Place, rules: List[Dict], feature_names: List[str]) -> Dict:
        """
        Create a DMN decision table from extracted rules.
        
        Args:
            place (PetriNet.Place): Place being processed
            rules (List[Dict]): Rules extracted from decision tree
            feature_names (List[str]): Feature names
            
        Returns:
            Dict: DMN decision table
        """
        # Extract unique features from all rules
        all_features = set()
        for rule in rules:
            for condition in rule['conditions']:
                all_features.add(condition['feature'])
        
        # Create DMN table structure
        dmn_table = {
            'id': f"decision_table_{place.name}",
            'name': f"Decision Table for {place.name}",
            'hit_policy': 'FIRST',  # Necessary element in DMN: First matching rule is applied
            'input_expressions': sorted(list(all_features)),
            'outputs': ['next activity decision'],
            'rules': []
        }
        
        # Convert each rule to DMN rule format
        for i, rule in enumerate(rules):
            dmn_rule = {
                'id': f"rule_{i+1}",
                'inputs': {},
                'output': rule['outcome'],
                'description': f"Support: {rule['support']:.2f}, Samples: {rule['samples']}"
            }
            
            # Process conditions
            for condition in rule['conditions']:
                feature = condition['feature']
                operator = condition['operator']
                value = condition['value']
                
                # Initialize this input if not present
                if feature not in dmn_rule['inputs']:
                    dmn_rule['inputs'][feature] = []
                
                # Add condition
                if operator == '==':
                    dmn_rule['inputs'][feature].append(value)
                elif operator == '!=':
                    dmn_rule['inputs'][feature].append(f"not({value})")
                elif operator == '<=':
                    dmn_rule['inputs'][feature].append(f"<= {value}")
                elif operator == '>':
                    dmn_rule['inputs'][feature].append(f"> {value}")
            
            # Add rule to table
            dmn_table['rules'].append(dmn_rule)
        
        return dmn_table
    
    def save_dmn_tables(self, output_dir: str = 'dmn_tables'):
        """
        Save the discovered DMN tables to JSON files.
        
        Args:
            output_dir (str): Directory to save the DMN tables to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for place, dmn_table in self.dmn_tables.items():
            filename = os.path.join(output_dir, f"dmn_table_{place.name}.json")
            with open(filename, 'w') as f:
                json.dump(dmn_table, f, indent=2)
            
            self._print_if_verbose(f"Saved DMN table for '{place.name}' to {filename}")

def discover_dmn(guard_ds_per_place: Dict[PetriNet.Place, pd.DataFrame],
                 min_samples_leaf: int = 5,
                 max_depth: int = 5,
                 min_rule_support: float = 0.05,
                 verbose: bool = True,
                 save_tables: bool = True,
                 output_dir: str = 'dmn_tables') -> Dict[PetriNet.Place, Dict]:
    """
    Discover DMN decision tables from feature tables.
    
    Args:
        guard_ds_per_place (Dict[PetriNet.Place, pd.DataFrame]): Dictionary mapping places to their feature tables
        min_samples_leaf (int): Minimum samples required to be at a leaf node
        max_depth (int): Maximum depth of the decision tree
        min_rule_support (float): Minimum support (fraction of samples) for a rule to be included
        verbose (bool): Whether to print verbose output
        save_tables (bool): Whether to save the DMN tables to files
        output_dir (str): Directory to save the DMN tables to
        
    Returns:
        Dict[PetriNet.Place, Dict]: Dictionary mapping places to their DMN decision tables
    """
    discoverer = DMNDiscovery(
        guard_ds_per_place=guard_ds_per_place,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        min_rule_support=min_rule_support,
        verbose=verbose
    )
    
    dmn_tables = discoverer.discover_all_dmn_tables()
    
    if save_tables:
        discoverer.save_dmn_tables(output_dir)
    
    return dmn_tables

if __name__ == "__main__":
    # Example usage
    from exdpn.util import import_log
    from experiments.main import main
    
    # Import log and extract features
    event_log = import_log('./datasets/p2p_base.xes')
    results = main(
        event_log=event_log,
        event_level_attributes=['item_category', 'item_id', 'item_amount', 'supplier', 'total_price'],
        verbose=True
    )
    
    _, _, _, guard_ds_per_place, _, _ = results
    
    # Discover DMN tables
    dmn_tables = discover_dmn(
        guard_ds_per_place=guard_ds_per_place,
        verbose=True
    )
    
    # Print summary
    print(f"\nDiscovered {len(dmn_tables)} DMN decision tables")
    for place, table in dmn_tables.items():
        print(f"  - Decision point '{place.name}': {len(table['rules'])} rules") 