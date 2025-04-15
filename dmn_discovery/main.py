import sys
import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from typing import List, Dict, Any, Tuple
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking
from exdpn.util import import_log
from exdpn.petri_net import get_petri_net
from exdpn.decisionpoints import find_decision_points
from exdpn.guard_datasets import extract_all_datasets
import pm4py.util.xes_constants as xes
from dmn_discovery import discover_dmn

def main(event_log: EventLog,
         miner_type: str = "IM",
         case_level_attributes: List[str] = [],
         event_level_attributes: List[str] = [],
         tail_length: int = 3,
         activityName_key: str = xes.DEFAULT_NAME_KEY,
         verbose: bool = True) -> Tuple[PetriNet, Marking, Marking, Dict, Dict, Dict, Dict]:
    """Processes an event log to extract a Petri net and train guards for decision points.

    Args:
        event_log (EventLog): The event log to be processed.
        miner_type (str, optional): Type of mining algorithm to use ("IM" or "AM"). Defaults to "IM".
        case_level_attributes (List[str], optional): Case-level attributes to consider. Defaults to [].
        event_level_attributes (List[str], optional): Event-level attributes to consider. Defaults to [].
        tail_length (int, optional): Number of preceding events to record. Defaults to 3.
        activityName_key (str, optional): Key for activity names in the event log. Defaults to xes.DEFAULT_NAME_KEY.
        ml_list (List[ML_Technique], optional): List of ML techniques to evaluate. Defaults to all implemented techniques.
        hyperparameters (Dict[ML_Technique, Dict[str, Any]], optional): Hyperparameters for ML techniques. Defaults to standard parameters.
        CV_splits (int, optional): Number of cross-validation folds. Defaults to 5.
        CV_shuffle (bool, optional): Whether to shuffle samples before splitting. Defaults to False.
        random_state (int, optional): Random state for reproducibility. Defaults to None.
        guard_threshold (float, optional): Performance threshold for adding guards. Defaults to 0.0.
        impute (bool, optional): Whether to impute missing values. Defaults to False.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.

    Returns:
        Tuple containing:
        - PetriNet: The mined or provided Petri net
        - Marking: Initial marking
        - Marking: Final marking
        - Dict: Guard datasets per place
        - Dict: Guard managers per place
        - Dict: Decision points in the Petri net
        - Dict: DMN decision tables per place
    """
    if verbose:
        print("-> Mining Petri net...", end="")
    
    # Get Petri net and markings
    petri_net, initial_marking, final_marking = get_petri_net(event_log, miner_type)
    
    if verbose:
        print("Done")
        print("-> Finding decision points...", end="")
    
    # Find decision points
    decision_points = find_decision_points(petri_net)
    
    if verbose:
        print("Done")
        print("-> Mining guard datasets...", end="")
    
    # Extract guard datasets
    guard_ds_per_place = extract_all_datasets(
        event_log, petri_net, initial_marking, final_marking,
        case_level_attributes, event_level_attributes,
        tail_length, activityName_key
    )
    
    if verbose:
        print("Done")
    
    # Create DMN tables from the guard datasets
    if verbose:
        print("\n-> Discovering DMN decision tables...")
    
    
    
    # Set DMN discovery parameters
    dmn_min_samples_leaf = 5
    dmn_max_depth = 4
    dmn_min_rule_support = 0.05
    
    # Discover DMN tables
    dmn_tables = discover_dmn(
        guard_ds_per_place=guard_ds_per_place,
        min_samples_leaf=dmn_min_samples_leaf,
        max_depth=dmn_max_depth,
        min_rule_support=dmn_min_rule_support,
        verbose=verbose,
        save_tables=True,
        output_dir='dmn_tables'
    )
    
    if verbose:
        print("\n-> DMN discovery summary:")
        for place, table in dmn_tables.items():
            print(f"  - Decision point '{place.name}': {len(table['rules'])} rules")
    
    return (
        petri_net,
        initial_marking,
        final_marking,
        guard_ds_per_place,
        decision_points,
        dmn_tables  # Add DMN tables to the returned results
    )

if __name__ == "__main__":
    # Example usage
    event_log = import_log('../datasets/p2p_base.xes')
    results = main(
        event_log=event_log,
        event_level_attributes=['item_category', 'item_id', 'item_amount', 'supplier', 'total_price'],
        verbose=True
    )
    
    # Unpack results
    petri_net, initial_marking, final_marking, guard_ds, decision_points, dmn_tables = results
    
    # Example of accessing results
    print(f"\nSummary of results:")
    print(f"Number of decision points: {len(decision_points)}")
    
    # Print some DMN table details
    print("\nDMN decision tables:")
    for place, table in dmn_tables.items():
        print(f"Decision point '{place.name}':")
        print(f"  - {len(table['rules'])} decision rules")
        print(f"  - Input features: {', '.join(table['input_expressions'])}")
        
        # Print a sample rule if available
        if table['rules']:
            print("  - Sample rule:")
            rule = table['rules'][0]
            print(f"    * Rule ID: {rule['id']}")
            print(f"    * Output: {rule['output']}")
            print(f"    * Description: {rule['description']}")
            
            # Print conditions
            print("    * Conditions:")
            for feature, conditions in rule['inputs'].items():
                print(f"      - {feature}: {', '.join([str(cond) for cond in conditions])}")
