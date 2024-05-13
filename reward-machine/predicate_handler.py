import enum
import numpy as np
from scipy.spatial import ConvexHull
from skspatial.objects import Vector
import tatsu
import tatsu.ast
import typing

from utils import extract_variable_type_mapping, extract_variables, extract_predicate_function_name, get_object_assignments, ast_cache_key, is_type_color_side_orientation, get_object_types, \
    _extract_object_limits, _object_corners, _point_in_object, _point_in_top_half, _object_location, FullState, ObjectState, AgentState, BuildingPseudoObject
from config import ALL_OBJECT_TYPES, UNITY_PSEUDO_OBJECTS, PseudoObject, DOOR_ID, WALL_ID, RUG_ID, RUG, ROOM_CENTER, ELIGILBLE_IN_OBJECT_IDS, ON_EXCLUDED_OBJECT_TYPES

# AgentState = typing.NewType('AgentState', typing.Dict[str, typing.Any])
# ObjectState = typing.NewType('ObjectState', typing.Union[str, typing.Any])
# ObjectOrPseudo = typing.Union[ObjectState, PseudoObject]  # TODO: figure out the syntax for this



AGENT_STATE_KEY = 'agentState'
AGENT_STATE_CHANGED_KEY = 'agentStateChanged'
N_OBJECTS_CHANGED_KEY = 'nObjectsChanged'
OBJECTS_KEY = 'objects'
ACTION_CHANGED_KEY = 'actionChanged'
ACTION_KEY = 'action'

ORIGINAL_INDEX_KEY = 'originalIndex'
OBJECT_ID_KEY = 'objectId'
OBJECT_NAME_KEY = 'name'


class PredicateHandlerPredicateNotImplemented(Exception):
    pass


class PredicateHandlerVariableNotInMapping(Exception):
    pass

PREDICATE_NOT_IMPLEMENTED_CACHE_VALUE = np.nan


class PredicateHandler:
    # Which field in the state to use as the index
    index_key: str
    # A cache of the first observed value for each object
    initial_state_cache: typing.Dict[str, typing.Union[ObjectState, AgentState, PseudoObject]]
    # Did the game just start?
    is_first_step: bool
    # Is the game ending after this step
    is_last_step: bool
    # Which field in each object to use as an id
    object_id_key: str
    # The cache from a string representation of the state X predicate X mapping to
    # the predicate's truth value in that state given the mapping.
    evaluation_cache: typing.Dict[str, typing.Union[bool, float]]
    # The last state the evaluation cache was updated for a given key.
    evaluation_cache_last_updated: typing.Dict[str, int]
    # A cache of the latest observed value for each object
    state_cache: typing.Dict[str, typing.Union[ObjectState, AgentState, PseudoObject]]
    # The last state the state cache was updated for
    state_cache_global_last_updated: int
    # The last state each object was updated for
    state_cache_object_last_updated: typing.Dict[str, int]

    def __init__(self, domain: str):
        self.domain = domain
        self._new_game()
        self.warned_objects = set()

    def _new_game(self):
        """
        Call when a new game is started to clear the cache.
        """
        self.evaluation_cache = {}
        self.evaluation_cache_last_updated = {}
        self.is_last_step = False
        self.is_first_step = True
        self.initial_state_cache = {}
        self.state_cache = {}
        self.state_cache_object_last_updated = {}
        self.state_cache_global_last_updated = -1

    def __call__(self, predicate: typing.Optional[tatsu.ast.AST], state: FullState,
        mapping: typing.Dict[str, str], force_evaluation: bool = False) -> bool:
        """
        The external API to the predicate handler.
        For now, implements the same logic as before, to make sure I separated it correctly from the `preference_handler`.
        After that, this will implement the caching logic.

        GD 2022-09-14: The data, as currently stored, saves delta updates of the state.
        This means that the truth value of a predicate with a particular assignment holds unitl
        there's information that merits updating it. This means that we should cache predicate
        evaluation results, update them when they return a non-None value, and return the cached result.

        GD 2022-09-29: We decided that since all external callers treat a None as a False, we might as well make it explicit here
        """
        try:
            pred_value = self._inner_call(predicate=predicate, state=state, mapping=mapping, force_evaluation=force_evaluation)
        except PredicateHandlerPredicateNotImplemented:
            pred_value = PREDICATE_NOT_IMPLEMENTED_CACHE_VALUE
        except PredicateHandlerVariableNotInMapping as e:
            missing_var = e.args[0]
            if missing_var not in self.warned_objects:
                print(f"Warning: variable {missing_var} not in mapping with keys {mapping.keys()}")
                self.warned_objects.add(missing_var)
            pred_value = False

        # if this reached all the way up here, we return True, since something not implemented could be True
        if pred_value == PREDICATE_NOT_IMPLEMENTED_CACHE_VALUE:
            return True

        return pred_value if pred_value is not None else False  # type: ignore

    def _inner_call(self, predicate: typing.Optional[tatsu.ast.AST], state: FullState,
        mapping: typing.Dict[str, str], force_evaluation: bool = False) -> typing.Optional[bool]:

        pred_variables = extract_variables(predicate)
        # the last if is to account for the case of a variable not already being in the mapping, if, e.g., it's quantified in a further down exists
        # e.g. in the expression "(forall (?d - (either dodgeball cube_block)) (game-optional (not (exists (?s - shelf) (on ?s ?d)))))"
        # when we reach the (game-optional ...) node, the variable ?s isn't in the mapping yet, and will be added later
        relevant_mapping = {}
        for var in pred_variables:
            if is_type_color_side_orientation(var):
                relevant_mapping[var] = var
            elif var in mapping:
                relevant_mapping[var] = mapping[var]
            elif not var.startswith('?'):
                raise PredicateHandlerVariableNotInMapping(var)


        relevant_mapping = {var: var if is_type_color_side_orientation(var) else mapping[var]
                            for var in pred_variables
                            if is_type_color_side_orientation(var) or var in mapping}

        predicate_key = "{0}_{1}".format(*ast_cache_key(predicate, relevant_mapping))
        state_index = state.original_index

        # If no time has passed since the last update, we know we can use the cached value
        if predicate_key in self.evaluation_cache_last_updated and self.evaluation_cache_last_updated[predicate_key] == state_index:
            return typing.cast(bool, self.evaluation_cache[predicate_key])

        # This shouldn't happen, but no reason not to check it anyhow
        if state_index > self.state_cache_global_last_updated:
            self.update_cache(state)

        try:
            current_state_value = self._inner_evaluate_predicate(predicate, state, relevant_mapping, force_evaluation)

        # if the exception propagated all the way here,
        except PredicateHandlerPredicateNotImplemented:
            # TODO: is this the right thing to do? I think so?
            current_state_value = PREDICATE_NOT_IMPLEMENTED_CACHE_VALUE

        if current_state_value is not None:
            self.evaluation_cache[predicate_key] = current_state_value
            self.evaluation_cache_last_updated[predicate_key] = state_index

        # Cache the value above so we don't go into trying to compute it again and if this is the case, reraise to propagate
        if current_state_value is PREDICATE_NOT_IMPLEMENTED_CACHE_VALUE:
            raise PredicateHandlerPredicateNotImplemented

        return typing.cast(bool, self.evaluation_cache.get(predicate_key, None))

    def update_cache(self, state: FullState):
        '''
        Update the cache if any objects / the agent are changed in the current state
        '''
        state_index = state.original_index
        if state_index > 0:
            self.is_first_step = False

        self.state_cache_global_last_updated = state_index
        for obj in state.objects:
            if obj.object_id not in self.initial_state_cache:
                self.initial_state_cache[obj.object_id] = obj
            else:
                obj = obj._replace(initial_rotation=self.initial_state_cache[obj.object_id].rotation)  # type: ignore

            self.state_cache[obj.object_id] = obj
            self.state_cache_object_last_updated[obj.object_id] = state_index

        if state.agent_state_changed:
            agent_state = typing.cast(AgentState, state.agent_state)
            self.state_cache[AGENT_STATE_KEY] = agent_state
            self.state_cache['agent'] = agent_state
            self.state_cache_object_last_updated[AGENT_STATE_KEY] = state_index
            self.state_cache_object_last_updated['agent'] = state_index

    def _evaluate_comparison_arg(self, comp_arg: typing.Union[float, tatsu.ast.AST], state: FullState,
                                 mapping: typing.Dict[str, str], force_evaluation: bool = False) -> typing.Tuple[typing.Optional[float], bool]:
        arg_implemented = True
        if isinstance(comp_arg, tatsu.ast.AST):
            try:
                # If this was passed a comparison arg, operate on the inner argument
                if comp_arg.parseinfo.rule == "comparison_arg":  # type: ignore
                    comp_arg = comp_arg["arg"]  # type: ignore

                # If it's an AST, it's a either a number or a function
                # if it's a nubmer, extract the value
                if comp_arg.parseinfo.rule == "comparison_arg_number_value":  # type: ignore
                    comp_arg = float(comp_arg["terminal"])  # type: ignore

                else:
                    comp_arg = self.evaluate_function(comp_arg, state, mapping, force_evaluation)  # type: ignore

                # If the function is undecidable with the current information, return None
                if comp_arg is None:
                    return None, arg_implemented

            except PredicateHandlerPredicateNotImplemented:
                arg_implemented = False
                comp_arg = float('-Inf')

        return float(comp_arg), arg_implemented  # type: ignore

    def _inner_evaluate_predicate(self, predicate: typing.Optional[tatsu.ast.AST], state: FullState,
        mapping: typing.Dict[str, str], force_evaluation: bool = False) -> typing.Optional[bool]:
        '''
        Given a predicate, a trajectory state, and an assignment of each of the predicate's
        arguments to specific objects in the state, returns the evaluation of the predicate

        GD: 2022-09-14: The logic in `__call__` relies on the assumption that if the predicate's
        value was not updated at this timestep, this function returns None, rather than False.
        This is to know when the cache should be updated.

        I should figure out how to implement this in a manner that's reasonable across all predicates,
        or maybe it's something the individual predicate handlers should do? Probably the latter.
        '''

        if predicate is None:
            return None

        predicate_rule = predicate.parseinfo.rule  # type: ignore

        if predicate_rule == "predicate":
            predicate_name = extract_predicate_function_name(predicate)  # type: ignore

            # Check for specific one-off predicates, like game-over, that can be evaluated without a mapping
            if predicate_name == "game_start":
                return self.is_first_step

            if predicate_name == "game_over":
                return self.is_last_step

            if predicate_name not in PREDICATE_LIBRARY:
                raise PredicateHandlerPredicateNotImplemented(predicate_name)

            # Obtain the functional representation of the base predicate
            predicate_fn = PREDICATE_LIBRARY[predicate_name]  # type: ignore

            # Determine the last step at which the predicate was evaluated
            predicate_key = "{0}_{1}".format(*ast_cache_key(predicate, mapping))
            predicate_mapping_last_updated = self.evaluation_cache_last_updated.get(predicate_key, -1)

            # Evaluate the predicate
            evaluation = predicate_fn(state, mapping, self.state_cache, self.state_cache_object_last_updated,
                                      predicate_mapping_last_updated, force_evaluation)

            return evaluation

        elif predicate_rule == "super_predicate":
            # No need to go back to __call__, there's nothing separate to cache here
            return self._inner_evaluate_predicate(predicate["pred"], state, mapping, force_evaluation)

        elif predicate_rule == "super_predicate_not":
            try:
                inner_pred_value = self._inner_call(predicate["not_args"], state, mapping, force_evaluation)
                return None if inner_pred_value is None else not inner_pred_value
            # Assume that if a predicate was not implemented, it could be either true or false, so treat it as True
            except PredicateHandlerPredicateNotImplemented:
                return True

        elif predicate_rule == "super_predicate_and":
            and_args = predicate["and_args"]
            if isinstance(and_args, tatsu.ast.AST):
                and_args = [and_args]

            # We can't speed this up as much by breaking out of the loop early when we encounter a False, because we don't
            # know if there would be an un-evaluable None later in the loop
            cur_truth_value = True
            any_arguments_implemented = False
            for sub in and_args:  # type: ignore
                try:
                    eval = self._inner_call(sub, state, mapping, force_evaluation)
                    any_arguments_implemented = True
                    if eval is None:
                        return None

                except PredicateHandlerPredicateNotImplemented:
                    continue

                cur_truth_value = cur_truth_value and eval

            # If all child predicates were not implemented, propagate the lack of implementation up
            if not any_arguments_implemented:
                raise PredicateHandlerPredicateNotImplemented('All children of AND were not implemented')

            return cur_truth_value

        elif predicate_rule == "super_predicate_or":
            or_args = predicate["or_args"]
            if isinstance(or_args, tatsu.ast.AST):
                or_args = [or_args]

            all_none = True
            any_arguments_implemented = False
            for sub in or_args:  # type: ignore
                try:
                    eval = self._inner_call(sub, state, mapping, force_evaluation) # outputs can be True, False, or None
                    any_arguments_implemented = True
                    if eval:
                        return True
                    elif eval == False:
                        all_none = False

                except PredicateHandlerPredicateNotImplemented:
                    continue

            # If at least one child was not implemented, propagate it up
            if not any_arguments_implemented:
                raise PredicateHandlerPredicateNotImplemented('All children of OR were not implemented')

            if all_none:
                return None

            return False

        elif predicate_rule == "super_predicate_exists":
            variable_type_mapping = extract_variable_type_mapping(predicate["exists_vars"]["variables"])  # type: ignore
            used_objects = list(mapping.values())
            object_assignments = get_object_assignments(self.domain, variable_type_mapping.values(), used_objects)  # type: ignore

            sub_mappings = [dict(zip(variable_type_mapping.keys(), object_assignment)) for object_assignment in object_assignments]

            all_none = True
            try:
                for sub_mapping in sub_mappings:
                    eval = self._inner_call(predicate["exists_args"], state, {**sub_mapping, **mapping}, force_evaluation)
                    if eval:
                        return True
                    elif eval == False:
                        all_none = False

            # If the only child (or their conjunction/disjunction) was not implemented, rereaise
            # The loop can live inside the try block, since it's the same argument over the entire loop
            except PredicateHandlerPredicateNotImplemented:
                raise PredicateHandlerPredicateNotImplemented('Children of EXISTS were not implemented')

            if all_none:
                return None

            return False

        elif predicate_rule == "super_predicate_forall":
            variable_type_mapping = extract_variable_type_mapping(predicate["forall_vars"]["variables"])  # type: ignore
            used_objects = list(mapping.values())
            object_assignments = get_object_assignments(self.domain, variable_type_mapping.values(), used_objects)  # type: ignore

            sub_mappings = [dict(zip(variable_type_mapping.keys(), object_assignment)) for object_assignment in object_assignments]

            cur_truth_value = True
            try:
                for sub_mapping in sub_mappings:
                    eval = self._inner_call(predicate["forall_args"], state, {**sub_mapping, **mapping}, force_evaluation)
                    if eval is None:
                        return None

                    cur_truth_value = cur_truth_value and eval

            # Same logic as with exists
            except PredicateHandlerPredicateNotImplemented:
                raise PredicateHandlerPredicateNotImplemented('Children of FORALL were not implemented')

            return cur_truth_value

        elif predicate_rule == "function_comparison":
            comp = typing.cast(tatsu.ast.AST, predicate["comp"])
            comparison_operator = comp["comp_op"]

            if "equal_comp_args" in comp:
                # For each comparison argument, evaluate it if it's a function or convert to an int if not
                args = comp["equal_comp_args"]
                if not isinstance(args, list):
                    args = [args]

                any_args_implemented = False
                all_args_implemented = True
                arg_values = []

                for arg in args:
                    arg_value, arg_implemented = self._evaluate_comparison_arg(arg["arg"],  # type: ignore
                                                                               state, mapping, force_evaluation)

                    if arg_value is None:
                        return None

                    any_args_implemented = any_args_implemented or arg_implemented
                    all_args_implemented = all_args_implemented and arg_implemented

                    if arg_implemented:
                        arg_values.append(arg_value)

                if not any_args_implemented:
                    raise PredicateHandlerPredicateNotImplemented('All arguments of function comparison were not implemented')

                if not all_args_implemented:
                    # check how many args we ended up with values for
                    if len(arg_values) == 1:
                        # Presumably the comparison is possible if we had implemented the other values, so return True
                        return True

                # At this point, we have at least one value, so we can compare them
                return np.all(np.isclose(arg_values, arg_values[0]))  # type: ignore

            else:
                # For each comparison argument, evaluate it if it's a function or convert to an int if not
                comp_arg_1, comp_arg_1_implemented = self._evaluate_comparison_arg(comp["arg_1"]["arg"],  # type: ignore
                                                                                   state, mapping, force_evaluation)

                if comp_arg_1 is None:
                    return None

                comp_arg_2, comp_arg_2_implemented = self._evaluate_comparison_arg(comp["arg_2"]["arg"],  # type: ignore
                                                                                   state, mapping, force_evaluation)

                if comp_arg_2 is None:
                    return None

                if not comp_arg_1_implemented and not comp_arg_1_implemented:
                    raise PredicateHandlerPredicateNotImplemented('Both arguments of function comparison were not implemented')

                # One was not implemented, so assume the function comparison could be True
                if comp_arg_1_implemented != comp_arg_2_implemented:
                    return True

                if comparison_operator == "=":
                    return comp_arg_1 == comp_arg_2
                elif comparison_operator == "<":
                    return comp_arg_1 < comp_arg_2
                elif comparison_operator == "<=":
                    return comp_arg_1 <= comp_arg_2
                elif comparison_operator == ">":
                    return comp_arg_1 > comp_arg_2
                elif comparison_operator == ">=":
                    return comp_arg_1 >= comp_arg_2
                else:
                    raise ValueError(f"Error: Unknown comparison operator '{comparison_operator}'")

        else:
            raise ValueError(f"Error: Unknown rule '{predicate_rule}'")

    def evaluate_function(self, function: typing.Optional[tatsu.ast.AST], state: FullState,
        mapping: typing.Dict[str, str], force_evaluation: bool = False) -> typing.Optional[float]:
        function_key = "{0}_{1}".format(*ast_cache_key(function, mapping))
        state_index = state.original_index

        # If no time has passed since the last update, we know we can use the cached value
        if function_key in self.evaluation_cache_last_updated and self.evaluation_cache_last_updated[function_key] == state_index:
            return self.evaluation_cache[function_key]

        # This shouldn't happen, but no reason not to check it anyhow
        if state_index > self.state_cache_global_last_updated:
            self.update_cache(state)

        current_state_value = self._inner_evaluate_function(function, state, mapping, force_evaluation)
        if current_state_value is not None:
            self.evaluation_cache[function_key] = current_state_value
            self.evaluation_cache_last_updated[function_key] = state_index

        return self.evaluation_cache.get(function_key, None)

    def _inner_evaluate_function(self, function: typing.Optional[tatsu.ast.AST], state: FullState,
        mapping: typing.Dict[str, str], force_evaluation: bool = False) -> typing.Optional[float]:

        if function is None:
            return None

        function_name = extract_predicate_function_name(function)  # type: ignore

        if function_name not in FUNCTION_LIBRARY:
            raise PredicateHandlerPredicateNotImplemented(function_name)

        # Obtain the functional representation of the function
        func = FUNCTION_LIBRARY[function_name]

        # Extract only the variables in the mapping relevant to this predicate
        relevant_mapping = {var: mapping[var] for var in extract_variables(function)}

        # Evaluate the function
        evaluation = func(state, relevant_mapping, self.state_cache, self.state_cache_object_last_updated, force_evaluation)

        # If the function is undecidable with the current information, return None
        if evaluation is None:
            return None

        return float(evaluation)


def mapping_objects_decorator(predicate_func: typing.Callable) -> typing.Callable:
    def wrapper(state: FullState, predicate_partial_mapping: typing.Dict[str, str], state_cache: typing.Dict[str, ObjectState],
        state_cache_last_updated: typing.Dict[str, int], predicate_mapping_last_updated: int, force_evaluation: bool = False) -> typing.Optional[bool]:

        agent_object = state.agent_state if state.agent_state_changed else state_cache[AGENT_STATE_KEY]

        # if there are no objects in the predicate mapping, then we can just evaluate the predicate
        if len(predicate_partial_mapping) == 0:
            return predicate_func(agent_object, [])

        # Otherwise, check if any of the relevant objects have changed in this state, excluding passed in types and colors
        mapping_items = predicate_partial_mapping.items()

        # The first time that we evaluate a predicate containing a PseudoObject (e.g. a wall), we need to
        # add it to the cache and mark it as updated for the current state in order to ensure that predicates
        # with only PseudoObjects in them are actually evaluated
        for var, obj in mapping_items:
            if obj in UNITY_PSEUDO_OBJECTS and obj not in state_cache:
                state_cache[obj] = UNITY_PSEUDO_OBJECTS[obj]  # type: ignore
                state_cache_last_updated[obj] = state.original_index

        any_object_not_in_cache = any(obj not in state_cache for var, obj in mapping_items if not is_type_color_side_orientation(var))

        # If any objects are not in the cache, we cannot evaluate the predidate
        if any_object_not_in_cache:
            if force_evaluation:
                raise ValueError(f'Attempted to force predicate evaluation while at least one object was not in the cache: {[(obj, obj in state_cache) for var, obj in mapping_items]}')
            return None

        any_objects_changed = any(state_cache_last_updated[obj] == state.original_index for var, obj in mapping_items if not is_type_color_side_orientation(var))

        latest_object_update = max(state_cache_last_updated[obj] for var, obj in mapping_items if not is_type_color_side_orientation(var))

        # If none of the objects have changed in the current step or since the last time we evaluated this predicate,
        # then return None unless force_evaluation is True
        if not (any_objects_changed or latest_object_update > predicate_mapping_last_updated or force_evaluation):
            return None

        mapping_objects = []
        for var, obj in mapping_items:
            mapping_objects.append(state_cache[obj] if not is_type_color_side_orientation(var) else obj)

        return predicate_func(agent_object, mapping_objects)

    return wrapper


# ====================================== PREDICATE DEFINITIONS ======================================


def _pred_generic_predicate_interface(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    """
    This isn't here to do anything useful -- it's just to demonstrate the interface that all predicates
    should follow.  The first argument should be the agent's state, and the second should be a list
    (potentially empty) of objects that are the arguments to this predicate.
    """
    raise NotImplementedError()


def _pred_above(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 2

    lower_object = objects[0]
    upper_object = objects[1]


    if isinstance(upper_object, AgentState):
        # The agent is only above something if the agent is on it (since the agent' can't jump)
        return _pred_on(agent, objects)

    if isinstance(upper_object, BuildingPseudoObject):
        # A building cannot be above the agent
        if isinstance(lower_object, AgentState):
            return False

        # A building is not above any object that's part of the building
        if lower_object.object_id in upper_object.building_objects:
            # Unless it's the bottom object in the building -- in which case the building is `on` it
            return _pred_on(agent, objects)

        # Otherwise, a building is above an object if an object in the building is above that object
        return any([_pred_above(agent, [lower_object, building_object])
            for building_object in upper_object.building_objects.values()
            ])

    upper_object_center = upper_object.bbox_center
    lower_object_center, lower_object_extents = lower_object.bbox_center, lower_object.bbox_extents
    return lower_object_center[0] - lower_object_extents[0] <= agent.position[0] <= lower_object_center[0] + lower_object_extents[0] and \
            lower_object_center[2] - lower_object_extents[2] <= agent.position[2] <= lower_object_center[2] + lower_object_extents[2] and \
            upper_object_center[1] > lower_object_center[1] + lower_object_extents[1]


def _pred_agent_crouches(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 0
    return agent.crouching


def _pred_agent_holds(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 1
    if isinstance(objects[0], PseudoObject) or isinstance(objects[0], AgentState):
        return False
    return agent.held_object == objects[0].object_id


def _pred_broken(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 1
    if isinstance(objects[0], PseudoObject) or isinstance(objects[0], AgentState):
        return False
    return objects[0].is_broken


EQUAL_POSITION_MARGIN = 0.1

def _pred_equal_x_position(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 2
    return np.isclose(objects[0].position[0], objects[1].position[0], atol=EQUAL_POSITION_MARGIN) or np.isclose(objects[0].bbox_center[0], objects[1].bbox_center[0], atol=EQUAL_POSITION_MARGIN)


def _pred_equal_z_position(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 2
    return np.isclose(objects[0].position[2], objects[1].position[2], atol=EQUAL_POSITION_MARGIN) or np.isclose(objects[0].bbox_center[2], objects[1].bbox_center[2], atol=EQUAL_POSITION_MARGIN)


def _pred_open(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 1
    if isinstance(objects[0], PseudoObject) or isinstance(objects[0], AgentState):
        return False
    return objects[0].is_open


def _pred_toggled_on(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 1
    if isinstance(objects[0], PseudoObject) or isinstance(objects[0], AgentState):
        return False
    return objects[0].is_toggled


def _pred_same_type(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject, str]]):
    assert len(objects) == 2

    # If the variable is an object, then we collect all of the types and meta-types that it belongs to. If
    # it's a type, then we just collect that type. The predicate is true if there is any overlap between
    # the two sets of types.

    if isinstance(objects[0], str):
        if objects[0] not in ALL_OBJECT_TYPES:
            raise ValueError(f"Invalid object type: {objects[0]} (may be a color)")

        object_1_types = set([objects[0]])

    elif isinstance(objects[0], AgentState):
        object_1_types = set(["agent"])

    else:
        object_1_types = get_object_types(objects[0])  # type: ignore

    if isinstance(objects[1], str):
        if objects[1] not in ALL_OBJECT_TYPES:
            raise ValueError(f"Invalid object type: {objects[1]} (may be a color)")

        object_2_types = set([objects[1]])

    elif isinstance(objects[0], AgentState):
        object_2_types = set(["agent"])

    else:
        object_2_types = get_object_types(objects[1])  # type: ignore

    type_intersection = object_1_types.intersection(object_2_types)

    return len(type_intersection) > 0


def _object_in_building(building: BuildingPseudoObject, other_object: ObjectState):
    return other_object.object_id in building.building_objects


IN_MARGIN = 0.075


def _pred_in(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]], allow_in_any_object: bool = False):
    assert len(objects) == 2

    # The agent cannot be a container or contained in another object
    if isinstance(objects[0], AgentState) or isinstance(objects[1], AgentState):
        return False

    first_pseudo = isinstance(objects[0], PseudoObject)
    second_pseudo = isinstance(objects[1], PseudoObject)

    if first_pseudo or second_pseudo:
        first_building = isinstance(objects[0], BuildingPseudoObject)
        second_building = isinstance(objects[1], BuildingPseudoObject)

        if first_building == second_building:
            return False  # a building cannot be inside another building, same holds for other pseudo objects

        if first_building:
            return _object_in_building(*objects)  # type: ignore

        # if the second object is a building, we continue to the standard implementation

    # Check that the first object is something that could contain other objects
    if not allow_in_any_object and objects[0].object_id not in ELIGILBLE_IN_OBJECT_IDS:
        return False

    outer_min_corner, outer_max_corner = _extract_object_limits(objects[0])
    inner_min_corner, inner_max_corner = _extract_object_limits(objects[1])

    return np.all(inner_min_corner >= outer_min_corner - IN_MARGIN) and np.all(inner_max_corner <= outer_max_corner + IN_MARGIN)


# TODO (GD): we should discuss what this threshold should be
IN_MOTION_ZERO_VELOCITY_THRESHOLD = 0.1

def _pred_in_motion(agent: AgentState, objects: typing.Sequence[ObjectState]):
    assert len(objects) == 1
    if isinstance(objects[0], PseudoObject):
        return False

    elif isinstance(objects[0], AgentState):
        return agent.last_movement_result # TODO: does this contain the information we want?

    return not (np.allclose(objects[0].velocity, 0, atol=IN_MOTION_ZERO_VELOCITY_THRESHOLD) and \
        np.allclose(objects[0].angular_velocity, 0, atol=IN_MOTION_ZERO_VELOCITY_THRESHOLD))


TOUCH_DISTANCE_THRESHOLD = 0.15


def _building_touch(agent: AgentState, building: BuildingPseudoObject, other_object: typing.Union[ObjectState, PseudoObject]):
    if not isinstance(other_object, AgentState) and other_object.object_id in building.building_objects:
        return False

    return any([_pred_touch(agent, [building_obj, other_object]) for building_obj in building.building_objects.values()])


def _pred_touch(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 2

    if isinstance(objects[0], str) or isinstance(objects[1], str):
        print(f'String objects in touch?! {objects}')

    first_pseudo = isinstance(objects[0], PseudoObject)
    first_agent = isinstance(objects[0], AgentState)
    second_pseudo = isinstance(objects[1], PseudoObject)
    second_agent = isinstance(objects[1], AgentState)

    # TODO (GD 2022-09-27): the logic here to decide which wall to attribute the collision here is incomoplete;
    # right now it assigns it to the nearest wall, but that could be incorrect, if the ball hit the wall at an angle
    # and traveled sufficiently far to be nearest another wall. This is a (literal) corner case, but one we probably
    # want to eventually resolve better, for example by simulating the ball back in time using the negative of its
    # velcoity and finding a wall it was most recently very close to?

    if first_pseudo and second_pseudo:
        first_building = isinstance(objects[0], BuildingPseudoObject)
        second_building = isinstance(objects[1], BuildingPseudoObject)
        if first_building == second_building:
            return False   # if both are buildings they would be merged; if neither, they wouldn't touch

        if first_building:
            return _building_touch(agent, objects[0], objects[1])  # type: ignore

        elif second_building:
            return _building_touch(agent, objects[1], objects[0])  # type: ignore

    elif first_pseudo:
        obj = typing.cast(ObjectState, objects[1])
        pseudo_obj = typing.cast(PseudoObject, objects[0])

        if isinstance(pseudo_obj, BuildingPseudoObject):
            return _building_touch(agent, pseudo_obj, obj)

        if second_agent:
            if pseudo_obj.object_id == DOOR_ID:
                return agent.touching_side and _pred_adjacent(agent, objects)

            if pseudo_obj.object_id == WALL_ID:
                return agent.touching_side and _pred_adjacent(agent, objects) \
                    and pseudo_obj is _find_nearest_pseudo_object_of_type(obj, pseudo_obj.object_type)

            if pseudo_obj.object_id == RUG_ID:
                return agent.touching_floor and _pred_on(agent, [pseudo_obj, obj]) \
                    and pseudo_obj is _find_nearest_pseudo_object_of_type(obj, pseudo_obj.object_type)

            # What does it mean to be touching the room center?
            if pseudo_obj.object_id == ROOM_CENTER:
                return False

            raise ValueError(f'Unknown pseudo object type: {pseudo_obj.object_id}')

        return any(identifier in obj.touching_objects for identifier in pseudo_obj.identifiers) and \
            pseudo_obj is _find_nearest_pseudo_object_of_type(obj, pseudo_obj.object_type)  # type: ignore

    elif second_pseudo:
        obj = typing.cast(ObjectState, objects[0])
        pseudo_obj = typing.cast(PseudoObject, objects[1])

        if isinstance(pseudo_obj, BuildingPseudoObject):
            return _building_touch(agent, pseudo_obj, obj)

        if first_agent:
            if pseudo_obj.object_id == DOOR_ID:
                return agent.touching_side and _pred_adjacent(agent, objects)

            if pseudo_obj.object_id == WALL_ID:
                return agent.touching_side and _pred_adjacent(agent, objects) \
                    and pseudo_obj is _find_nearest_pseudo_object_of_type(obj, pseudo_obj.object_type)

            if pseudo_obj.object_id == RUG_ID:
                return agent.touching_floor and _pred_on(agent, [pseudo_obj, obj]) \
                    and pseudo_obj is _find_nearest_pseudo_object_of_type(obj, pseudo_obj.object_type)

            # What does it mean to be touching the room center?
            if pseudo_obj.object_id == ROOM_CENTER:
                return False

            raise ValueError(f'Unknown pseudo object type: {pseudo_obj.object_id}')

        return any(identifier in obj.touching_objects for identifier in pseudo_obj.identifiers) and \
            pseudo_obj is _find_nearest_pseudo_object_of_type(obj, pseudo_obj.object_type)  # type: ignore

    # gd1279: the agent appears as `FPSController` in the touching objects of the object it is touching
    elif first_agent:
        return 'FPSController' in objects[1].touching_objects  # type: ignore

    elif second_agent:
        return 'FPSController' in objects[0].touching_objects  # type: ignore

    else:
        return objects[1].object_id in objects[0].touching_objects or objects[0].object_id in objects[1].touching_objects  # type: ignore


ON_DISTANCE_THRESHOLD = 0.01

def _pred_on(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 2

    lower_object = objects[0]
    upper_object = objects[1]

    # Special case: the agent can only be 'on' the floor or the rug
    if isinstance(upper_object, AgentState):
        if 'Rug' not in lower_object.object_id and 'Floor' not in lower_object.object_id:
            return False

        rug_pseudo_object = UNITY_PSEUDO_OBJECTS[RUG]

        rug_position, rug_extents = rug_pseudo_object.position, rug_pseudo_object.bbox_extents
        agent_on_rug = rug_position[0] - rug_extents[0] <= agent.position[0] <= rug_position[0] + rug_extents[0] and \
            rug_position[2] - rug_extents[2] <= agent.position[2] <= rug_position[2] + rug_extents[2]

        return agent_on_rug if 'Rug' in lower_object.object_id else not agent_on_rug

    # Special case: nothing can ever be 'on' the agent
    if isinstance(lower_object, AgentState):
        return False

    # Checking for objects where it shouldn't happen unless they weirdly clip in the environment
    if lower_object.object_type in ON_EXCLUDED_OBJECT_TYPES:
        return False

    if _pred_touch(agent, objects):
        upper_object_bbox_center = upper_object.bbox_center
        upper_object_bbox_extents = upper_object.bbox_extents

        # Project a point slightly below the bottom center / corners of the upper object
        upper_object_corners = _object_corners(upper_object)

        test_points = [corner - np.array([0, upper_object_bbox_extents[1] + ON_DISTANCE_THRESHOLD, 0])  # type: ignore
                       for corner in upper_object_corners]
        test_points.append(upper_object_bbox_center - np.array([0, upper_object_bbox_extents[1] + ON_DISTANCE_THRESHOLD, 0]))

        # Due to bounding box weirdness, we also check to see if the center of the upper object is contained in the bottom's
        # bounding box. Enforcing that the objects are touching should make sure that we don't have any errors where floating
        # objects are considered on top of other objects, but we should keep an eye on this for any weird behavior that crops up
        # test_points += upper_object_corners

        objects_on = any([_point_in_top_half(test_point, lower_object) for test_point in test_points])

        # object 1 is on object 0 if they're touching and object 1 is above object 0
        # or if they're touching and object 1 is contained withint object 0?
        return objects_on or _pred_in(agent, objects, allow_in_any_object=True)

    elif isinstance(upper_object, BuildingPseudoObject):
        # A building can also be on an object if that object is (a) in the building
        if lower_object.object_id not in upper_object.building_objects:
            return False

        # (b) that object is not on any other object in the building
        if any([_pred_on(agent, [building_object, lower_object])
            for building_object in upper_object.building_objects.values()
            if building_object.object_id != lower_object.object_id]):
            return False

        # and (c) at least one object in the building is on that object
        return any([_pred_on(agent, [lower_object, building_object])
            for building_object in upper_object.building_objects.values()
            if building_object.object_id != lower_object.object_id])

    elif isinstance(lower_object, BuildingPseudoObject):
        # An object is on a building if that object is (a) in the building
        if upper_object.object_id not in lower_object.building_objects:
            return False

        # (b) no other object in the building is on that object
        if any([_pred_on(agent, [upper_object, building_object])
            for building_object in lower_object.building_objects.values()
            if building_object.object_id != upper_object.object_id]):
            return False

        # and (c) it is on at least one object in the building
        return any([_pred_on(agent, [building_object, upper_object])
            for building_object in lower_object.building_objects.values()
            if building_object.object_id != upper_object.object_id])

    return False

NEAR_DISTANCE_THRESHOLD = 0.75


def _pred_near(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 2
    return _func_distance(agent, objects) <= NEAR_DISTANCE_THRESHOLD


ADJACENT_DISTANCE_THRESHOLD = 0.2
OVERLAP_GRACE = 0.01
OBJECT_SIZE_SCALING = 1.2

def _pred_adjacent(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 2

    object_1_min, object_1_max = _extract_object_limits(objects[0])
    object_2_min, object_2_max = _extract_object_limits(objects[1])

    # Determine if there is overlap for each of the dimensions
    x_overlap = (object_1_min[0] - OVERLAP_GRACE <= object_2_max[0] + OVERLAP_GRACE) and \
                (object_2_min[0] - OVERLAP_GRACE <= object_1_max[0] + OVERLAP_GRACE)

    y_overlap = (object_1_min[1] - OVERLAP_GRACE <= object_2_max[1] + OVERLAP_GRACE) and \
                (object_2_min[1] - OVERLAP_GRACE <= object_1_max[1] + OVERLAP_GRACE)

    z_overlap = (object_1_min[2] - OVERLAP_GRACE <= object_2_max[2] + OVERLAP_GRACE) and \
                (object_2_min[2] - OVERLAP_GRACE <= object_1_max[2] + OVERLAP_GRACE)

    # Two objects can only be adjacent if there is some overlap in their y extents
    if not y_overlap:
        return False

    # Measures the minimum distance between any pair of parallel sides between the two objects
    x_displacement = min(abs(object_1_min[0] - object_2_max[0]), abs(object_2_min[0] - object_1_max[0]),
                         abs(object_1_min[0] - object_2_min[0]), abs(object_2_max[0] - object_1_max[0]))

    z_displacement = min(abs(object_1_min[2] - object_2_max[2]), abs(object_2_min[2] - object_1_max[2]),
                         abs(object_1_min[2] - object_2_min[2]), abs(object_2_max[2] - object_1_max[2]))

    object_dist = _func_distance(agent, objects)

    # Intuition: an object is not adjacent to another if it's more than (some scaling >= 1) times its own size away from it.
    # Since adjacency is symmetric, we'll use the larger of the two objects to determine the threshold distance. We'll also
    # first try determing an object's size by taking the average of its two dimensions (x and z)

    object_1_size = (objects[0].bbox_extents[0] + objects[0].bbox_extents[2]) # don't need to divide by 2 since the extent is half the size
    object_2_size = (objects[1].bbox_extents[0] + objects[1].bbox_extents[2])

    # Can try average of the two objects' sizes, or just use the larger of the two. Can also try various scaling factors
    size = OBJECT_SIZE_SCALING * (object_1_size + object_2_size) / 2

    threshold_dist = min(ADJACENT_DISTANCE_THRESHOLD, size)

    # Adjacency for a given side (e.g. X) is determined by whether the displacement is below the threshold and the objects overlap
    # in the opposite side extents (e.g. Z)
    adjacent_by_x = x_displacement <= threshold_dist and z_overlap
    adjacent_by_z = z_displacement <= threshold_dist and x_overlap
    adjacent_by_dist = object_dist <= threshold_dist

    # if isinstance(objects[1], PseudoObject) and objects[1].name == "south_wall":
    #     print("\n==========\nObject:", objects[0].name)
    #     print("Object position:", objects[0].bbox_center)
    #     print("Object extents:", objects[0].bbox_extents)

    #     print("Wall position:", objects[1].bbox_center)
    #     print("Wall extents:", objects[1].bbox_extents)

    #     print("\nThreshold:", threshold_dist)
    #     print("Adjacent by x:", x_displacement, z_overlap)
    #     print("Adjacent by z:", z_displacement, x_overlap)
    #     print("Adjacent by dist:", adjacent_by_dist)

    return adjacent_by_dist or adjacent_by_x or adjacent_by_z

def _pred_between(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 3

    object_1_bottom_corners = _object_corners(objects[0], y_pos="bottom")
    object_1_top_corners = _object_corners(objects[0], y_pos="top")

    object_2_bottom_corners = _object_corners(objects[2], y_pos="bottom")
    object_2_top_corners = _object_corners(objects[2], y_pos="top")

    test_position = _object_location(objects[1])

    # An object is between two others if its center position is contained in the convex hull formed by the vertices of the
    # others. We can test this by seeing if that the test position is *not* among the vertices of the hull

    hull = ConvexHull(np.concatenate([object_1_bottom_corners, object_1_top_corners, object_2_bottom_corners, object_2_top_corners,
                                      np.array(test_position).reshape(1, -1)]))

    # The test point is always at index 16
    return 16 not in hull.vertices

def _pred_faces(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 2

    caster, target = objects

    # For simplicitly, we zero out the y component in each vector
    projection = np.array([0, 2])

    caster_pos = _object_location(caster)[projection]
    caster_rotation =  caster.camera_rotation_euler_angles if isinstance(caster, AgentState) else caster.rotation
    caster_facing = caster_rotation[projection]

    target_pos = _object_location(target)[projection]
    target_corners = _object_corners(target, y_pos=0)  # type: ignore

    caster_to_target = Vector(target_pos - caster_pos)
    caster_to_corners = [Vector(corner[projection] - caster_pos) for corner in target_corners]

    angle_to_corners = [caster_to_target.angle_signed(to_corner) for to_corner in caster_to_corners]
    min_corner_angle, max_corner_angle = min(angle_to_corners), max(angle_to_corners)  # type: ignore

    # Clearly this won't work, because the caster's rotation is not the same as its facing direction
    angle_to_facing = caster_to_target.angle_signed(caster_facing)

    # print("\n" + "=" * 100)
    # print("Caster:", caster.object_id)
    # print("\tRotation:", caster.rotation)
    # print("Target:", target.object_id)
    # print("Angle to corners:", angle_to_corners)
    # print("Angle to facing:", angle_to_facing)

    return min_corner_angle <= angle_to_facing <= max_corner_angle


OBJECT_ORIENTATION_ANGLE_MARGIN = 15


def _pred_object_orientation(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 2

    obj = objects[0]
    orientation = objects[1]

    if isinstance(obj, AgentState):
        return orientation == 'upright'

    if isinstance(obj, PseudoObject) or obj.initial_rotation is None:
        return orientation == 'upright'

    rotation_diff = 180 - np.abs(np.abs(obj.rotation - obj.initial_rotation) - 180)
    sideways = (abs(rotation_diff[0] - 90) <= OBJECT_ORIENTATION_ANGLE_MARGIN) or (abs(rotation_diff[2] - 90) <= OBJECT_ORIENTATION_ANGLE_MARGIN)
    upside_down = ((abs(rotation_diff[0] - 180) <= OBJECT_ORIENTATION_ANGLE_MARGIN) != (abs(rotation_diff[2] - 180) <= OBJECT_ORIENTATION_ANGLE_MARGIN)) and not sideways
    upright = (abs(rotation_diff[0]) <= OBJECT_ORIENTATION_ANGLE_MARGIN) or (abs(rotation_diff[2]) <= OBJECT_ORIENTATION_ANGLE_MARGIN)

    if orientation == 'upright':
        return upright

    if orientation == 'sideways':
        return sideways

    if orientation == 'upside_down':
        return upside_down

    if orientation == 'diagonal':
        return not (upright or sideways or upside_down)

    raise ValueError(f'Invalid orientation: {orientation}')



# ====================================== FUNCTION DEFINITIONS =======================================


def _find_nearest_pseudo_object_of_type(object: ObjectState, object_type: str):
    """
    Finds the pseudo object in the sequence that is closest to the object.
    """
    filtered_pseudo_objects = [obj for obj in UNITY_PSEUDO_OBJECTS.values() if obj.object_type == object_type]
    if len(filtered_pseudo_objects) == 1:
        return filtered_pseudo_objects[0]

    distances = [_distance_object_pseudo_object(object, pseudo_object) for pseudo_object in filtered_pseudo_objects]
    return filtered_pseudo_objects[np.argmin(distances)]


def _get_pseudo_object_relevant_distance_dimension_index(pseudo_object: PseudoObject):
    if np.allclose(pseudo_object.rotation[1], 0):
        return 2

    if np.allclose(pseudo_object.rotation[1], 90):
        return 0

    raise NotImplemented(f'Cannot compute distance between object and pseudo object with rotation {pseudo_object.rotation}')


def _distance_object_pseudo_object(object: ObjectState, pseudo_object: PseudoObject):
    distance_dimension = _get_pseudo_object_relevant_distance_dimension_index(pseudo_object)
    return np.linalg.norm(_object_location(object)[distance_dimension] - _object_location(pseudo_object)[distance_dimension])


def _func_distance(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 2

    first_pseudo = isinstance(objects[0], PseudoObject)
    second_pseudo = isinstance(objects[1], PseudoObject)

    if first_pseudo and second_pseudo:
        # handled identically to the case where neither is a pseudo object
        pass
    elif first_pseudo:
        return _distance_object_pseudo_object(objects[1], objects[0])  # type: ignore
    elif second_pseudo:
        return _distance_object_pseudo_object(objects[0], objects[1])  # type: ignore


    return np.linalg.norm(_object_location(objects[0]) - _object_location(objects[1]))


    # TODO: do we want to use the position? Or the bounding box?


def _func_building_size(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 1
    assert isinstance(objects[0], BuildingPseudoObject)

    return len(objects[0].building_objects)



# ================================= EXTRACTING LIBRARIES FROM LOCALS() ==================================


PREDICATE_PREFIX = '_pred_'

PREDICATE_LIBRARY = {local_key.replace(PREDICATE_PREFIX, ''): mapping_objects_decorator(local_val_pred)
    for local_key, local_val_pred in locals().items()
    if local_key.startswith(PREDICATE_PREFIX)
}

PREDICATE_LIBRARY_RAW = {local_key.replace(PREDICATE_PREFIX, ''): local_val_pred
    for local_key, local_val_pred in locals().items()
    if local_key.startswith(PREDICATE_PREFIX)
}

FUNCTION_PREFIX = '_func_'

FUNCTION_LIBRARY = {local_key.replace(FUNCTION_PREFIX, ''): mapping_objects_decorator(local_val_func)
    for local_key, local_val_func in locals().items()
    if local_key.startswith(FUNCTION_PREFIX)
}
