"""
.. include:: ./../../docs/_templates/md/guard_datasets/guard_datasets.md

"""

from exdpn.decisionpoints import find_decision_points

from pandas import DataFrame, MultiIndex

from typing import Dict, List, Tuple, Union, Any
import numpy as np

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.util import xes_constants as xes


def extract_all_datasets(
    log: EventLog,
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    case_level_attributes: List[str] = [],
    event_level_attributes: List[str] = [],
    tail_length: int = 3,
    activityName_key: str = xes.DEFAULT_NAME_KEY,
    places: List[PetriNet.Place] = None,
    padding: Any = "#"
) -> Dict[PetriNet.Place, DataFrame]:
    """Extracts a dataset for each decision point using token-based replay. For each instance of this decision found in the log, the following data is extracted:
    1. The specified case-level attributes of the case
    2. The specified event-level attributes of the last event of the case before this decision is made
    3. The acitivities executed in the events contained in the `tail_length` events before the decision
    4. The transition which is chosen (the *target* class)

    Args:
        log (EventLog): The event log to extract the data from.
        net (PetriNet): The Petri net on which the token-based replay will be performed and on which the decision points.
        initial_marking (Marking): The initial marking of the Petri net.
        final_marking (Marking): The final marking of the Petri net.
        case_level_attributes (List[str], optional): The list of attributes to be extracted on a case-level. Defaults to empty list.
        event_level_attributes (List[str], optional): The list of attributes to be extracted on an event-level. Defaults to empty list.
        tail_length (int, optional): The number of preceding events to record. Defaults to 3.
        activityName_key (str, optional): The key of the activity name in the event log. Defaults to `pm4py.util.xes_constants.DEFAULT_NAME_KEY` ("concept:name").
        places (List[Place], optional): The list of places to extract datasets for. If not present, all decision points are regarded.
        padding (Any, optional): The padding to be used when the tail goes over beginning of the case. Defaults to "#".

    Returns:
        Dict[Place, DataFrame]: The dictionary mapping places in the Petri net to their corresponding dataset.

    Examples:
        
        >>> from exdpn.util import import_log
        >>> from exdpn.petri_net import get_petri_net
        >>> from exdpn.guard_datasets import extract_all_datasets
        >>> event_log = import_log('./datasets/p2p_base.xes')
        >>> pn, im, fm = get_petri_net(event_log)
        >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
        ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
        ...                                       activityName_key = "concept:name")

        
        .. include:: ../../docs/_templates/md/example-end.md
    """

    # Get list of places and mapping which transitions they correspond to
    target_transitions = find_decision_points(net)
    if places is None:
        # Use all decision points as places
        places = list(target_transitions.keys())
    else:
        # Use only the transitions corresponding to decision points of the list of places
        target_transitions = {
            place: target_transitions[place] for place in places
        }

    # Compute Token-Based Replay
    replay = _compute_replay(log, net, initial_marking, final_marking, stop_immediately_unfit=False, activityName_key=activityName_key, show_progress_bar=False)
    ## Extract a dataset for each place ##
    datasets = dict()
    for place in places:
        datasets[place] = extract_dataset_for_place(
            place, target_transitions, log, replay, case_level_attributes, event_level_attributes, tail_length, activityName_key, padding)
    return datasets


def _compute_replay(log: EventLog, net: PetriNet, initial_marking: Marking, final_marking: Marking, stop_immediately_unfit: bool = False, activityName_key: str = xes.DEFAULT_NAME_KEY, show_progress_bar: bool = False) -> Dict[str, Any]:
    """Wrapper for PM4Py's token-based replay function.

    Args:
        log (EventLog): The event log to use for Replay.
        net (PetriNet): The Petri net to replay on.
        initial_marking (Marking): The initial Marking of the Petri net.
        final_marking (Marking): The final Marking of the Petri net.
        stop_immediately_unfit (bool, optional): Whether to stop the replay as soon as a trace is unfit. Useful for recognizing unfinished cases. Defaults to False.
        activityName_key (str, optional): The key of the activity name in the event log. Defaults to `pm4py.util.xes_constants.DEFAULT_NAME_KEY` ("concept:name").
        show_progress_bar (bool, optional): Whether or not to show a progress bar. Defaults to False.

    Returns:
        The token-based replay results.

    """
    variant = token_replay.Variants.TOKEN_REPLAY
    replay_params = {
        variant.value.Parameters.SHOW_PROGRESS_BAR: show_progress_bar,
        variant.value.Parameters.ACTIVITY_KEY: activityName_key,
        variant.value.Parameters.STOP_IMMEDIATELY_UNFIT: stop_immediately_unfit
    }
    return token_replay.apply(log, net, initial_marking, final_marking, variant=variant, parameters=replay_params)


def extract_dataset_for_place(
    place: PetriNet.Place,
    target_transitions: Dict[PetriNet.Place, PetriNet.Transition],
    log: EventLog,
    replay: Union[List[Dict[str, Any]], Tuple[PetriNet, Marking, Marking]],
    case_level_attributes: List[str] = [],
    event_level_attributes: List[str] = [],
    tail_length: int = 3,
    activityName_key: str = xes.DEFAULT_NAME_KEY,
    padding: Any = "#"
) -> DataFrame:
    """Extracts the dataset for a single place using token-based replay. For each instance of this decision found in the log, the following data is extracted:
    1. The specified case-level attributes of the case
    2. The specified event-level attributes of the last event of the case before this decision is made
    3. The acitivities executed in the events contained in the `tail_length` events before the decision
    4. The transition which is chosen (the *target* class)


    Args:
        place (PetriNet.Place): The place for which to extract the data.
        target_transitions (Dict[PetriNet.Place, PetriNet.Transition]): The transitions which have an input arc from this place.
        log (EventLog): The Event Log from which to extract the data.
        replay (List[Dict[str, Any]] | Tuple[PetriNet, Marking, Marking]): Either the token-based replay computed by PM4Py, or the net which to use to compute the replay.
        case_level_attributes (List[str], optional): The list of attributes to be extracted on a case-level. Defaults to empty list.
        event_level_attributes (List[str], optional): The list of attributes to be extracted on an event-level. Defaults to empty list.
        tail_length (int, optional): The number of preceding events to record. Defaults to 3.
        activityName_key (str, optional): The key of the activity name in the event log. Defaults to `pm4py.util.xes_constants.DEFAULT_NAME_KEY` ("concept:name").
        padding (Any, optional): The padding to be used when the tail goes over beginning of the case. Defaults to "#".

    Returns:
        DataFrame: The guard-dataset extracted for the decision point at `place`.

    Raises:
        Exception: If the default case ID key defined by the XES standard ("concept:name") is not among the case-level attributes.

    """

    # Compute replay if necessary
    if type(replay) is tuple:
        net, im, fm = replay
        replay = _compute_replay(log, net, im, fm, stop_immediately_unfit=False, activityName_key=activityName_key, show_progress_bar=False)

    # Extract the data for the place
    instances = []
    indices = []
    for idx, trace_replay in enumerate(replay):
        # Track how often this decision is made in the trace, for unique Dataframe index
        decision_repetition = 0
        if not trace_replay["trace_is_fit"]:
            # Skip non-fitting traces
            continue
        # Track index of current event because invisible transitions can be present
        event_index = 0
        for transition in trace_replay["activated_transitions"]:

            if transition in target_transitions[place]:
                # Extract Case-Level Attributes
                case = log[idx]
                case_attr_values = [case.attributes.get(
                    attr, np.nan) for attr in case_level_attributes]

                if event_index <= 0:
                    # There is no "previous event", so we cannot collect this info
                    event_attr_values = [np.nan] * len(event_level_attributes)
                else:
                    # Get the values of the event level attribute
                    last_event = case[event_index-1]
                    event_attr_values = [last_event.get(
                        attr, np.nan) for attr in event_level_attributes]

                # Finally, extract recent activities
                tail_activities = []
                for i in range(1, tail_length+1):
                    if event_index-i >= 0:
                        tail_activities.append(
                            case[event_index-i].get(activityName_key, ""))
                    else:
                        tail_activities.append(padding)

                # This instance record  now descibes the decision situation
                instance = case_attr_values + event_attr_values + \
                    tail_activities + [transition]
                instances.append(instance)
                # Give this index a unique index
                if xes.DEFAULT_TRACEID_KEY not in case.attributes:
                    raise Exception(
                        f"A case in the Event Log Object has no caseid (No case attribute {xes.DEFAULT_TRACEID_KEY})")
                else:
                    indices.append(
                        (case.attributes[xes.DEFAULT_TRACEID_KEY], decision_repetition))
                decision_repetition += 1

                # Dont't count silent transitions
            if transition.label is not None:
                event_index += 1
    return DataFrame(
        instances,
        columns=["case::" + attr for attr in case_level_attributes] + 
                ["event::"+ attr for attr in event_level_attributes] + 
                [f"tail::prev{i}" for i in range(1, tail_length+1)] + 
                ["target"],
        index=MultiIndex.from_tuples(
            indices, names=[xes.DEFAULT_TRACEID_KEY, "decision_repetiton"])
    )

def extract_current_decisions(
    log: EventLog,
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    case_level_attributes: List[str] = [],
    event_level_attributes: List[str] = [],
    tail_length: int = 3,
    activityName_key: str = xes.DEFAULT_NAME_KEY,
    places: List[PetriNet.Place] = None,
    padding: Any = "#"
) -> Dict[PetriNet.Place, DataFrame]:
    """Extracts the current decisions of an event log. \
        These are all current decisons of unfinished cases. \
        Unfinished cases are identified as the unfit traces which can perfectly be replayed on the model, but do not reach a final marking in token-based replay. \
        Current decisions of an unfit trace arise at those places which have enabled transitions in the token based replay-marking \
        and correspond to the latest instance of such a trace.

    Args:
        log (EventLog): The event log to extract the data from.
        net (PetriNet): The Petri net on which the token-based replay will be performed and on which the decision points.
        initial_marking (Marking): The initial marking of the Petri net.
        final_marking (Marking): The final marking of the Petri net.
        case_level_attributes (List[str], optional): The list of attributes to be extracted on a case-level. Defaults to empty list.
        event_level_attributes (List[str], optional): The list of attributes to be extracted on an event-level. Defaults to empty list.
        tail_length (int, optional): The number of preceding events to record the activity of. Defaults to 3.
        activityName_key (str, optional): The key of the activity name in the event log. Defaults to `pm4py.util.xes_constants.DEFAULT_NAME_KEY` ("concept:name").
        places (List[Place], optional): The list of places to extract datasets for. If not present, all decision points are regarded.
        padding (Any, optional): The padding to be used when the tail goes over beginning of the case. Defaults to "#".
    Returns:
        Dict[Place, DataFrame]: The dictionary mapping places from `places` to their corresponding dataset of current decisions.
    """

    replay = _compute_replay(log, net, initial_marking, final_marking, stop_immediately_unfit=True, activityName_key=activityName_key, show_progress_bar=False)


    target_transitions = find_decision_points(net)
    if places is None:
        # Use all decision points as places
        places = list(target_transitions.keys())
    else:
        # Use only the transitions corresponding to decision points of the list of places
        target_transitions = {
            place: target_transitions[place] for place in places
        }

    datasets = {}

    for place in places:
        instances = []
        indices = []

        for idx, trace in enumerate(log):
            trace_replay = replay[idx]

            # A trace is incomplete if it is not fitting on the model, but it is a prefix of the language of the model
            activated_transitions_non_silent = [transition for transition in trace_replay["activated_transitions"] if transition.label is not None] # Skip silent transitions in the model for this check
            trace_is_incomplete = (
                (not trace_replay["trace_is_fit"]) and # The trace is unfit
                all(event[activityName_key] == transition.label for event, transition in zip(trace, activated_transitions_non_silent)) # But it can be perfectly replayed on the model as a prefix of the language of the model
            )
            if not trace_is_incomplete:
                # Skip fitting traces
                continue

            if trace_replay["enabled_transitions_in_marking"].intersection(target_transitions[place]):
                # If a transition is enabled that is the "output" of some place (this place)
                index, instance = extract_current_decision_for_trace(
                    trace, case_level_attributes, event_level_attributes, tail_length, activityName_key, padding)
                
                instances.append(instance)
                indices.append(index)
        
        if len(instances) != 0:
            datasets[place] = DataFrame(
                instances,
                columns=["case::" + attr for attr in case_level_attributes] + 
                        ["event::"+ attr for attr in event_level_attributes] + 
                        [f"tail::prev{i}" for i in range(1, tail_length+1)] + 
                        ["target"],
                index=indices
            )
            datasets[place].index.name = xes.DEFAULT_TRACEID_KEY

    return datasets

def extract_current_decision_for_trace(
    trace: Trace,
    case_level_attributes: List[str] = [],
    event_level_attributes: List[str] = [],
    tail_length: int = 3,
    activityName_key: str = xes.DEFAULT_NAME_KEY,
    padding: Any = "#",
) -> Tuple[Any, List[Any]]:
    """Extract prediction information for a trace.

    Args:
        trace (Trace): The trace to extract the data from.
        case_level_attributes (List[str], optional): The case-level attributes to extract from the last event of the case. Defaults to the empty list.
        event_level_attributes (List[str], optional): The event-level attributes to extract from the last event of the case. Defaults to the empty list.
        tail_length (int, optional): The number of preceding events to record the activity of. Defaults to 3.
        activityName_key (str, optional):  The key of the activity name in the event log. Defaults to `pm4py.util.xes_constants.DEFAULT_NAME_KEY` ("concept:name").
        padding (Any, optional): The padding to be used when the tail goes over beginning of the case. Defaults to "#".

    Raises:
        ValueError: If a case in the event log has no case-id, an error is raised.

    Returns:
        Tuple[Any, List[Any]]: A tuple containing the index for the dataframe (the case-ID) and the list of extracted values.
    """
    case_attr_values = [trace.attributes.get(
        attr, np.nan) for attr in case_level_attributes]

    if len(trace) == 0:
        # There is no "previous event", so we cannot collect this info
        event_attr_values = [np.nan] * len(event_level_attributes)
    else:
        # Get the values of the event level attribute
        last_event = trace[-1]
        event_attr_values = [
            last_event.get(
                attr,
                np.nan
            )
            for attr in event_level_attributes
        ]

        # Finally, extract recent activities
        tail_activities = []
        for i in range(1, tail_length+1):
            if len(trace)-i >= 0:
                tail_activities.append(
                    trace[-i].get(activityName_key, ""))
            else:
                tail_activities.append(padding)

        # This instance record  now descibes the decision situation
        instance = case_attr_values + event_attr_values + \
            tail_activities + [padding]
        # Give this index a unique index
        if xes.DEFAULT_TRACEID_KEY not in trace.attributes:
            raise ValueError(
                f"A case in the Event Log Object has no caseid (No case attribute {xes.DEFAULT_TRACEID_KEY})")
        else:
            index = trace.attributes[xes.DEFAULT_TRACEID_KEY]

    return index, instance 



# tests implemented examples
if __name__ == "__main__":
    import doctest
    doctest.testmod()
# run python .\exdpn\guard_datasets\data_extraction.py from eXdpn file
