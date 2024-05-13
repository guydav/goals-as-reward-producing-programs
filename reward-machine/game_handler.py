from collections import defaultdict
import itertools
import numpy as np
import tatsu
import tatsu.ast
import tatsu.grammars
import typing

from math import prod

from config import OBJECTS_BY_ROOM_AND_TYPE, NAMED_OBJECTS, SPECIFIC_NAMED_OBJECTS_BY_ROOM
from preference_handler import PreferenceHandler, PreferenceSatisfaction
from predicate_handler import PredicateHandler
from building_handler import BuildingHandler
from utils import extract_variable_type_mapping, get_object_assignments, ast_cache_key, FullState
from predicate_handler import _pred_in_motion


DEFAULT_GRAMMAR_PATH = "./dsl/dsl.ebnf"

class ObjectMove(typing.NamedTuple):
    time: int
    pos: np.ndarray

class GameHandler():
    domain_name: str
    game_name: str
    setup: typing.Optional[tatsu.ast.AST]
    setup_met : bool
    preferences: typing.List[tatsu.ast.AST]
    terminal: typing.Optional[tatsu.ast.AST]
    scoring: typing.Optional[tatsu.ast.AST]
    game_ast: tatsu.ast.AST
    predicate_handler: PredicateHandler
    preference_satisfactions: typing.Dict[str, typing.List[PreferenceSatisfaction]]
    building_handler: BuildingHandler

    def __init__(self, game: typing.Union[str, tatsu.ast.AST], grammar_path: str = DEFAULT_GRAMMAR_PATH, verbose: bool = False,
                 force_domain: str = '', ignore_preference_names: typing.Optional[typing.List[str]] = None, ignore_setup: bool = False):
        self.verbose = verbose
        self.domain_name = force_domain
        if ignore_preference_names is None:
            ignore_preference_names = []
        self.ignore_preference_names = ignore_preference_names
        self.ignore_setup = ignore_setup

        self.game_name = ''
        self.setup = None
        self.setup_met = False
        self.game_optional_cache = set()
        self.game_conserved_cache = {} # maps from a setup condition to the mapping that satisfied it first
        self.preferences = []
        self.terminal = None
        self.scoring = None


        if isinstance(game, str):
            grammar = open(grammar_path).read()
            self.grammar_parser = typing.cast(tatsu.grammars.Grammar, tatsu.compile(grammar))
            self.game_ast = self.grammar_parser.parse(game)  # type: ignore
        else:
            self.game_ast = game

        self._extract_game_info(self.game_ast)  # type: ignore

        if not self.domain_name:
            raise ValueError("Error: Failed to extract domain from game specification")

        self.building_handler = BuildingHandler(self.domain_name)
        self.predicate_handler = PredicateHandler(self.domain_name)

        # Maps from an object ID to a list of the time indexes in which it moves and the corresponding position at that time
        self.object_movements = defaultdict(list)
        self.cur_step = 0
        self.initial_update_complete = False

        # Maps from each preference name to the PreferenceHandler (or list of PreferenceHandlers) that will
        # evaluate that preference
        self.preference_handlers = {}

        # Maps from each preference name to a list of satisfaction data. Each entry in the list
        # is a namedtuple of type PreferenceSatisfcation (defined in preference_handler.py)
        self.preference_satisfactions = {}

        for preference in self.preferences:
            pref_def = typing.cast(tatsu.ast.AST, preference["definition"])
            rule = pref_def["parseinfo"].rule   # type: ignore

            # A preference definition expands into either (forall <variable_list> <preference>) or <preference>
            if rule == "preference":
                name = typing.cast(str, pref_def["pref_name"])

                pref_handler = PreferenceHandler(pref_def, self.predicate_handler, self.domain_name, verbose=self.verbose)
                self.preference_handlers[name] = pref_handler
                self.preference_satisfactions[name] = []
                if self.verbose: print(f"Successfully constructed PreferenceHandler for '{name}'")

            # This case handles externall forall preferences
            elif rule == "pref_forall":

                forall_vars = pref_def["forall_vars"]
                forall_pref = pref_def["forall_pref"]

                variable_type_mapping = extract_variable_type_mapping(forall_vars["variables"])  # type: ignore

                sub_preferences = forall_pref["preferences"] # type: ignore
                if isinstance(sub_preferences, tatsu.ast.AST):
                    sub_preferences = [sub_preferences]

                for sub_preference in sub_preferences:
                    name = typing.cast(str, sub_preference["pref_name"])

                    pref_handler = PreferenceHandler(sub_preference, self.predicate_handler, self.domain_name,
                        additional_variable_mapping=variable_type_mapping, verbose=self.verbose)
                    self.preference_handlers[name] = pref_handler
                    self.preference_satisfactions[name] = []
                    if self.verbose: print(f"Successfully constructed PreferenceHandler for '{name}'")

    def _extract_game_info(self, ast: typing.Union[list, tuple, tatsu.ast.AST]):
        '''
        Recursively extract the game's name, domain, setup, preferences, terminal conditions, and
        scoring (if they exist)
        '''
        if isinstance(ast, tuple) or isinstance(ast, list):
            for item in ast:
                self._extract_game_info(item)

        elif isinstance(ast, tatsu.ast.AST):
            rule = ast["parseinfo"].rule  # type: ignore
            if rule == "game_def":
                self.game_name = typing.cast(str, ast["game_name"])

            elif rule == "domain_def":
                if not self.domain_name:  # in case I force it in the constructor
                    self.domain_name = ast["domain_name"].split("-")[0]  # type: ignore
                    if self.domain_name not in OBJECTS_BY_ROOM_AND_TYPE:
                        raise ValueError(f"Error: Domain '{self.domain_name}' not supported (not found in the keys of OBJECTS_BY_ROOM_AND_TYPE: {list(OBJECTS_BY_ROOM_AND_TYPE.keys())}")

            elif rule == "setup" and not self.ignore_setup:
                self.setup = ast["setup"]

            elif rule == "preferences":
                prefs = ast["preferences"] # type: ignore
                # Handle games with single preference
                if isinstance(prefs, tatsu.ast.AST):
                    prefs = [prefs]

                self.preferences = [pref for pref in prefs if pref["pref_name"] not in self.ignore_preference_names]  # type: ignore

            elif rule == "terminal":
                self.terminal = ast["terminal"]

            elif rule == "scoring_expr":
                self.scoring = ast

    def _extract_scoring_preferences(self, scoring_expression: typing.Union[list, tuple, tatsu.ast.AST]) -> typing.List[str]:
        '''
        Extract all the preferences (by name) that are used in the scoring expression
        '''

        if isinstance(scoring_expression, tuple) or isinstance(scoring_expression, list):
            return sum([self._extract_scoring_preferences(item) for item in scoring_expression], [])

        elif isinstance(scoring_expression, tatsu.ast.AST):
            rule = scoring_expression["parseinfo"].rule  # type: ignore
            if rule == "pref_name_and_types":
                return [scoring_expression["pref_name"]]  # type: ignore

            return sum([self._extract_scoring_preferences(item) for item in scoring_expression.values()], [])

        return []


    def process(self, state: FullState, is_final: bool, debug: bool = False,
        debug_building_handler: bool = False, debug_preference_handlers: bool = False, ignore_terminals: bool = False) -> typing.Optional[float]:
        '''
        Process a state in a game trajectory by passing it to each of the relevant PreferenceHandlers. If the state is
        the last one in the trajectory or the terminal conditions are met, then we also do scoring
        '''

        self.cur_step += 1

        self.predicate_handler.update_cache(state)
        self.building_handler.process(state, debug=debug or debug_building_handler)

        # Every named object will exist only once in the room, so we can just directly use index 0
        default_mapping = {}
        for obj in NAMED_OBJECTS:
            if obj in OBJECTS_BY_ROOM_AND_TYPE[self.domain_name]:
                default_mapping[obj] = OBJECTS_BY_ROOM_AND_TYPE[self.domain_name][obj][0]

            elif obj in SPECIFIC_NAMED_OBJECTS_BY_ROOM[self.domain_name]:
                default_mapping[obj] = SPECIFIC_NAMED_OBJECTS_BY_ROOM[self.domain_name][obj][0]

            else:
                raise ValueError(f"Error: Could not find object id for named object '{obj}'")

        # Update with specifically named objects that exist once in this domain
        default_mapping.update({obj: SPECIFIC_NAMED_OBJECTS_BY_ROOM[self.domain_name][obj][0] for obj in SPECIFIC_NAMED_OBJECTS_BY_ROOM[self.domain_name]
                                if obj not in default_mapping and len(SPECIFIC_NAMED_OBJECTS_BY_ROOM[self.domain_name][obj]) == 1})
        setup = self.evaluate_setup(self.setup, state, default_mapping)
        self.setup_met = self.setup_met or setup
        if not setup:

            # Manually advance 'cur_step' for each PreferenceHandler, since their process() methods aren't being called
            for handler in self.preference_handlers.values():
                handler.cur_step += 1

            return

        # Check for object updates. If an object moves, then the current time is added to its list of motion times
        if state.n_objects_changed > 0:
            for obj in state.objects:
                if _pred_in_motion(state.agent_state, [obj]):  # type: ignore
                    self.object_movements[obj.object_id].append(ObjectMove(self.cur_step, obj.position))

        # When we get our first full state update, we treat every object as though it moved. This is so that we can
        # evaluate whether an object is stationary even in situations where it never moves (see count_unique_positions)
        if state.n_objects_changed > 15 and not self.initial_update_complete:
            for obj in state.objects:
                self.object_movements[obj.object_id].append(ObjectMove(self.cur_step, obj.position))
            self.initial_update_complete = True

        # The game is in its last step if the terminal conditions are met or if the trajectory is over. We pass this
        # value to the PredicateHandler so that it can evaluate game-over
        is_last_step = is_final or (not ignore_terminals and self.evaluate_terminals(self.terminal))
        self.predicate_handler.is_last_step = is_last_step

        for preference_name, handlers in self.preference_handlers.items():
            if isinstance(handlers, PreferenceHandler):
                satisfactions = handlers.process(state, is_last_step, debug=debug or debug_preference_handlers)
                self.preference_satisfactions[preference_name] += satisfactions

            elif isinstance(handlers, list): # TODO: is this case safe to remove?
                pass

        if is_last_step:
            score = self.score(self.scoring)

        else:
            score = None

        return score

    def evaluate_setup(self, setup_expression: typing.Optional[tatsu.ast.AST], state: FullState,
                       mapping: typing.Dict[str, str], called_from_forall=False) -> bool:
        '''
        Determine whether the setup conditions of the game have been met. The setup conditions
        of a game are met if all of the game-optional expressions have evaluated to True at least
        once in the past and if all of the game-conserved expressions currently evaluate to true
        '''

        if setup_expression is None:
            return True

        rule = setup_expression["parseinfo"].rule  # type: ignore

        if rule == "setup":
            return self.evaluate_setup(setup_expression["setup"], state, mapping, called_from_forall)

        elif rule == "setup_statement":
            return self.evaluate_setup(setup_expression["statement"], state, mapping, called_from_forall)

        elif rule == "super_predicate":
            # TODO: @gdrtodd, do you remember why the force_evaluation flag is needed here?
            evaluation = self.predicate_handler(setup_expression, state, mapping, force_evaluation=self.cur_step > 100)

            return evaluation

        elif rule == "setup_not":
            inner_value = self.evaluate_setup(setup_expression["not_args"], state, mapping, called_from_forall)
            return not inner_value

        elif rule == "setup_and":
            if isinstance(setup_expression["and_args"], tatsu.ast.AST):
                return self.evaluate_setup(setup_expression["and_args"], state, mapping)

            for sub in setup_expression["and_args"]:  # type: ignore
                if not self.evaluate_setup(sub, state, mapping, called_from_forall):
                    return False

            return True

        elif rule == "setup_or":
            if isinstance(setup_expression["or_args"], tatsu.ast.AST):
                return self.evaluate_setup(setup_expression["or_args"], state, mapping)

            for sub in setup_expression["or_args"]:  # type: ignore
                if self.evaluate_setup(sub, state, mapping, called_from_forall):
                    return True

            return False

        elif rule == "setup_exists":
            variable_type_mapping = extract_variable_type_mapping(setup_expression["exists_vars"]["variables"])  # type: ignore
            object_assignments = get_object_assignments(self.domain_name, variable_type_mapping.values())  # type: ignore

            sub_mappings = [dict(zip(variable_type_mapping.keys(), object_assignment)) for object_assignment in object_assignments]

            for sub_mapping in sub_mappings:
                if self.evaluate_setup(setup_expression["exists_args"], state, {**sub_mapping, **mapping}, called_from_forall):
                    return True

            return False

        elif rule == "setup_forall":
            variable_type_mapping = extract_variable_type_mapping(setup_expression["forall_vars"]["variables"])  # type: ignore
            object_assignments = get_object_assignments(self.domain_name, variable_type_mapping.values()) # type: ignore

            sub_mappings = [dict(zip(variable_type_mapping.keys(), object_assignment)) for object_assignment in object_assignments]

            for sub_mapping in sub_mappings:
                if not self.evaluate_setup(setup_expression["forall_args"], state, {**sub_mapping, **mapping}, called_from_forall=True):
                    return False
            return True

        elif rule == "setup_game_optional":
            # Once the game-optional condition has been satisfied once, we no longer need to evaluate it
            cache_key = "{0}_{1}".format(*ast_cache_key(setup_expression["optional_pred"], mapping))
            if cache_key in self.game_optional_cache:
                return True

            evaluation = self.evaluate_setup(setup_expression["optional_pred"], state, mapping, called_from_forall)
            if evaluation:
                self.game_optional_cache.add(cache_key)

            return evaluation

        elif rule == "setup_game_conserved":
            # For a game-conserved condition, we store the first object assignment that satisfies it
            # and ensure that the condition is satisfied *by those objects* in all future states
            expr_str, mapping_str = ast_cache_key(setup_expression["conserved_pred"], mapping)

            evaluation = self.evaluate_setup(setup_expression["conserved_pred"], state, mapping, called_from_forall)

            # We only lock in the object assignment if we're not being called from a forall
            if called_from_forall:
                return evaluation

            # If we've satisfied the condition for the first time, store the mapping in the cache
            if evaluation and expr_str not in self.game_conserved_cache:
                self.game_conserved_cache[expr_str] = mapping_str

            return evaluation and self.game_conserved_cache.get(expr_str) == mapping_str

        else:
            raise ValueError(f"Error: Unknown setup rule '{rule}'")


    def evaluate_terminals(self, terminal_expression: typing.Optional[tatsu.ast.AST]) -> bool:
        '''
        Determine whether the terminal conditions of the game have been met
        '''
        if terminal_expression is None:
            return False

        rule = terminal_expression["parseinfo"].rule  # type: ignore

        if rule == "terminal":
            return self.evaluate_terminals(terminal_expression["terminal"])

        elif rule == "terminal_not":
            inner_value = self.evaluate_terminals(terminal_expression["not_args"])

            return not inner_value

        elif rule == "terminal_and":
            inner_values = [self.evaluate_terminals(sub) for sub in terminal_expression["and_args"]]  # type: ignore

            return all(inner_values)

        elif rule == "terminal_or":
            inner_values = [self.evaluate_terminals(sub) for sub in terminal_expression["or_args"]]   # type: ignore

            return any(inner_values)

        # Interestingly, in the grammar a terminal comparison can only over have 2 arguments (i.e. there can be no
        # (= arg1 arg2 arg3)), so this makes extracting the expressions a bit more straightforward
        elif rule == "terminal_comp":
            terminal_expression = typing.cast(tatsu.ast.AST, terminal_expression["comp"])
            comparison_operator = terminal_expression["op"]

            expr_1 = self.score(terminal_expression["expr_1"]["expr"]) # type: ignore
            expr_2 = self.score(terminal_expression["expr_2"]["expr"]) # type: ignore

            if comparison_operator == "=":
                return expr_1 == expr_2
            elif comparison_operator == "<":
                return expr_1 < expr_2
            elif comparison_operator == "<=":
                return expr_1 <= expr_2
            elif comparison_operator == ">":
                return expr_1 > expr_2
            elif comparison_operator == ">=":
                return expr_1 >= expr_2
            else:
                raise ValueError(f"Error: Unknown comparison operator '{comparison_operator}'")

        else:
            raise ValueError(f"Error: Unknown terminal rule '{rule}'")

    def _extract_name_and_types(self, scoring_expression: tatsu.ast.AST) -> typing.Tuple[str, typing.Optional[typing.Sequence[str]]]:
        '''
        Helper function to extract the name of the preference being scored, as well as any of the object types that have been
        passed to it using the ":" syntax
        '''
        name_and_types = typing.cast(tatsu.ast.AST, scoring_expression["name_and_types"])
        preference_name = name_and_types["pref_name"]

        object_types = name_and_types["object_types"]
        if object_types is not None:
            if not isinstance(object_types, list):
                object_types = [object_types]

            object_types = [t.type_name.terminal for t in object_types]

        return str(preference_name), object_types


    def _filter_satisfactions(self, preference_name: str, object_types: typing.Optional[typing.Sequence[str]],
                              external_mapping: typing.List[tuple]):
        '''
        Filter the list of satisfactions of a given preference using two criteria:
        1. object_types: used with an external forall and the ":" syntax, filters satisfactions to those where the objects
           used to satisfy the external forall are of the given *types*

        2. external_mapping: used with an external forall, filters satisfactions to those where the objects used to satisfy
           the external forall are of the given *object IDs*
        '''

        if object_types is None and external_mapping == []:
            return self.preference_satisfactions[preference_name]

        pref_handler = self.preference_handlers[preference_name]
        satisfactions = []

        for potential_sat in self.preference_satisfactions[preference_name]:
            mapping = potential_sat.mapping
            acceptable_sat = True

            for idx, variable in enumerate(pref_handler.additional_variable_mapping.keys()):

                # Check to see if the object type matches
                if object_types is not None:
                    specififed_object = mapping[variable]
                    object_type = object_types[idx]

                    # Added the first check to account for moving games between rooms
                    if object_type not in OBJECTS_BY_ROOM_AND_TYPE[self.domain_name] or specififed_object not in OBJECTS_BY_ROOM_AND_TYPE[self.domain_name][object_type]:
                        acceptable_sat = False
                        break

                # Check to see if the object ID matches
                if external_mapping:
                    if mapping[variable] != external_mapping[idx]:
                        acceptable_sat = False
                        break

            if acceptable_sat:
                satisfactions.append(potential_sat)

        return satisfactions

    def score(self, scoring_expression: typing.Union[str, tatsu.ast.AST, None],
              external_mapping: typing.List[tuple] = []) -> float:
        '''
        Determine the score of the current trajectory using the given scoring expression
        '''
        if scoring_expression is None:
            return 0.0

        # TODO: is the only situation in which we'll directly score a string?
        if isinstance(scoring_expression, str):
            return float(scoring_expression)

        rule = scoring_expression["parseinfo"].rule  # type: ignore

        if rule in ("scoring_expr", "scoring_expr_or_number"):
            return self.score(scoring_expression["expr"], external_mapping)

        elif rule in ("comparison_arg_number_value", "time_number_value", "score_number_value", "pref_count_number_value", "scoring_number_value"):
            return float(scoring_expression["terminal"]) # type: ignore

            # At this point, the rule is positive_number
            return float(number["terminal"]) * sign  # type: ignore

        elif rule == "scoring_multi_expr":
            # Multi-expression operators are either addition (+) or multiplication (*)
            operator = scoring_expression["op"]

            # Despite being multi-expression operators, they can still accept one or more arguments
            expressions = scoring_expression["expr"]

            if isinstance(expressions, tatsu.ast.AST):
                return self.score(expressions, external_mapping)

            elif isinstance(expressions, list):
                if operator == "+":
                    return sum([self.score(expression, external_mapping) for expression in expressions])

                elif operator == "*":
                    return prod([self.score(expression, external_mapping) for expression in expressions])

        elif rule == "scoring_binary_expr":
            # Binary expression operators are either subtraction (-) or division (/)
            operator = scoring_expression["op"]

            expr_1 = scoring_expression["expr_1"]
            expr_2 = scoring_expression["expr_2"]

            score_1 = self.score(expr_1, external_mapping)
            score_2 = self.score(expr_2, external_mapping)

            if operator == "-":
                return score_1 - score_2
            elif operator == "/":
                return score_1 / score_2

        elif rule == "scoring_neg_expr":
            return - self.score(scoring_expression["expr"], external_mapping)

        elif rule == "scoring_comparison":
            comp_expr = typing.cast(tatsu.ast.AST, scoring_expression["comp"])
            comparison_operator = comp_expr["op"]

            # In this case, we know that the operator is = and that we have more than 2 comparison arguments,
            # so we just determine whether all arguments evaluate to the same value
            if comparison_operator is None:
                expressions = comp_expr["expr"]
                evaluations = [self.score(expr, external_mapping) for expr in expressions]  # type: ignore

                return float(evaluations.count(evaluations[0]) == len(evaluations))

            # Otherwise, there will be exactly two comparison arguments and we can compare them normally
            else:
                expr_1 = comp_expr["expr_1"]
                expr_2 = comp_expr["expr_2"]

                score_1 = self.score(expr_1, external_mapping)
                score_2 = self.score(expr_2, external_mapping)

                if comparison_operator == "=":
                    return score_1 == score_2
                elif comparison_operator == "<":
                    return score_1 < score_2
                elif comparison_operator == "<=":
                    return score_1 <= score_2
                elif comparison_operator == ">":
                    return score_1 > score_2
                elif comparison_operator == ">=":
                    return score_1 >= score_2
                else:
                    raise ValueError(f"Error: Unknown comparison operator '{comparison_operator}'")

        elif rule == "preference_eval":
            return self.score(scoring_expression["count_method"], external_mapping)

        elif rule == "scoring_external_maximize":
            maximized_preferences = self._extract_scoring_preferences(scoring_expression)

            # Make sure that at least one of the predicates is under an external forall, and that the predicates
            # in total are not under more than one external forall
            external_quantifications = [self.preference_handlers[pref_name].additional_variable_mapping for pref_name in maximized_preferences
                                        if self.preference_handlers[pref_name].additional_variable_mapping != {}]

            if len(external_quantifications) == 0:
                raise ValueError("Error: No external quantification found for maximization")

            for quant in external_quantifications:
                if any([quant != q for q in external_quantifications]):
                    raise ValueError("Error: All predicates in an external maximize must be under the same external forall")

            external_quant = external_quantifications[0]

            # We extract the mappings that are actually used in the satisfactions of the preferences under maximization
            all_satisfactions = sum([self.preference_satisfactions[pref_name] for pref_name in maximized_preferences
                                     if self.preference_handlers[pref_name].additional_variable_mapping != {}], [])

            # If there aren't any satisfactions, then we can just return 0
            if len(all_satisfactions) == 0:
                return 0.0

            # Each entry in the set is a tuple of the objects used to satisfy the external mapping for at least one satisfaction.
            # Because the mapping is an OrderedDict, the order of the objects in the tuple is the same as the order of the variables
            # in the external forall
            used_external_mappings = set([tuple([satisfaction.mapping[key] for key in external_quant]) for satisfaction in all_satisfactions])

            # Return the score computed using the external mapping that maximizes the score
            return max([self.score(scoring_expression["scoring_expr"], external_mapping) for external_mapping in used_external_mappings])  # type: ignore

        elif rule == "scoring_external_minimize":
            # Identical except for a single line to scoring_external_maximize, so see above for comments
            minimized_preferences = self._extract_scoring_preferences(scoring_expression)

            external_quantifications = [self.preference_handlers[pref_name].additional_variable_mapping for pref_name in minimized_preferences
                                        if self.preference_handlers[pref_name].additional_variable_mapping != {}]

            if len(external_quantifications) == 0:
                raise ValueError("Error: No external quantification found for minimization")

            for quant in external_quantifications:
                if any([quant != q for q in external_quantifications]):
                    raise ValueError("Error: All predicates in an external minimize must be under the same external forall")

            external_quant = external_quantifications[0]

            all_satisfactions = sum([self.preference_satisfactions[pref_name] for pref_name in minimized_preferences
                                     if self.preference_handlers[pref_name].additional_variable_mapping != {}], [])

            if len(all_satisfactions) == 0:
                return 0.0

            used_external_mappings = set([tuple([satisfaction.mapping[key] for key in external_quant]) for satisfaction in all_satisfactions])

            return min([self.score(scoring_expression["scoring_expr"], external_mapping) for external_mapping in used_external_mappings])  # type: ignore

        # Count the number of satisfactions of the given preference that don't overlap in both
        # (a) the mapping of variables to objects
        # (b) the temporal states involved
        elif rule == "count":
            preference_name, object_types = self._extract_name_and_types(scoring_expression)
            satisfactions = self._filter_satisfactions(preference_name, object_types, external_mapping)

            # Group the satisfactions by their mappings. Within each group, ensure there are no state overlaps and
            # count the total number of satisfactions that satisfy those criteria
            count = 0

            keyfunc = lambda satisfaction: "_".join(satisfaction.mapping.values())
            for key, group in itertools.groupby(sorted(satisfactions, key=keyfunc), keyfunc):
                group = list(sorted(group, key=lambda satisfaction: satisfaction.end))

                prev_end = -1
                for mapping, start, end, measures in group:
                    if start >= prev_end:
                        prev_end = end
                        count += 1

            return count

        # Count the largest set of satisfactions that share a mapping and overlap in their start / ends (solve with DP?)
        elif rule == "count_overlapping":
            preference_name, object_types = self._extract_name_and_types(scoring_expression)
            satisfactions = self._filter_satisfactions(preference_name, object_types, external_mapping)

            # Determine the largest set of satisfactions that overlap by sorting the start / end times. Whenever we
            # encounter a new start time, we increment a counter. Whenever we encounter an end time, we decrement the
            # counter. The maximum value of the counter at any point in time is the size of the largest set of satisfactions
            # that overlap

            cur_max = 0
            cur_count = 0
            starts_and_ends = []

            for satisfaction in satisfactions:
                starts_and_ends.append((satisfaction.start, "start"))
                starts_and_ends.append((satisfaction.end, "terminal")) # because of sorting, the string here needs to be "terminal" and not "end"

            for time, type in sorted(starts_and_ends):
                if type == "start":
                    cur_count += 1
                elif type == "terminal":
                    cur_count -= 1

                cur_max = max(cur_max, cur_count)

            return cur_max

        # Count whether the preference has been satisfied at all
        elif rule == "count_once":
            preference_name, object_types = self._extract_name_and_types(scoring_expression)

            satisfactions = self._filter_satisfactions(preference_name, object_types, external_mapping)

            return 1 if len(satisfactions) > 0 else 0

        # Count the number of satisfactions of the given preference that use distinct variable mappings
        elif rule == "count_once_per_objects":
            preference_name, object_types = self._extract_name_and_types(scoring_expression)

            satisfactions = self._filter_satisfactions(preference_name, object_types, external_mapping)

            count = 0

            # We consider two mappings to be distinct if at least one variable is mapped to a different object,
            # so we call (?b: ball_1, ?x - block_1) and (?b: block_1, ?x - ball_1) distinct
            keyfunc = lambda satisfaction: "_".join(satisfaction.mapping.values())
            for key, group in itertools.groupby(sorted(satisfactions, key=keyfunc), keyfunc):
                count += 1

            return count

        # For each satisfaction (see count above), sum the value of the measurement
        elif rule == "count_measure":
            preference_name, object_types = self._extract_name_and_types(scoring_expression)

            satisfactions = self._filter_satisfactions(preference_name, object_types, external_mapping)

            count = 0

            keyfunc = lambda satisfaction: "_".join(satisfaction.mapping.values())
            for key, group in itertools.groupby(sorted(satisfactions, key=keyfunc), keyfunc):
                group = list(sorted(group, key=lambda satisfaction: satisfaction.end))

                prev_end = -1
                for mapping, start, end, measures in group:
                    if start >= prev_end:
                        prev_end = end
                        if measures is not None:
                            count += list(measures.values())[0] # TODO: will we only ever have one measurement per preference?

            return count

        elif rule == "count_unique_positions":
            preference_name, object_types = self._extract_name_and_types(scoring_expression)
            satisfactions = self._filter_satisfactions(preference_name, object_types, external_mapping)

            # Maps from an object ID to the list of positions that it has remained stationary at throughout a satisfaction
            used_positions = defaultdict(list)
            UNIQUE_POSITION_TOL = 0.01

            count = 0

            # TODO: do we need to filter out overlapping satisfactions?

            # For each satisfaction, we need to determine which of its quantified objects are stationary within the
            # duration of the satisfaction
            for satisfaction in satisfactions:
                all_unique = True
                encountered_stationary = False

                for obj in satisfaction.mapping.values():
                    if not any([satisfaction.start < time < satisfaction.end for time, pos in self.object_movements[obj]]):
                        # Obtains the position of the object at its move closest to (but before) the start of the satisfaction
                        last_position = max(filter(lambda move: move.time < satisfaction.start, self.object_movements[obj]),
                                            key=lambda move: move.time).pos

                        # Check whether this position is too close to any previously used unique positions
                        if any([np.linalg.norm(last_position - pos) < UNIQUE_POSITION_TOL for pos in used_positions[obj]]):
                            all_unique = False
                            break

                        used_positions[obj].append(last_position)
                        encountered_stationary = True

                # If we reach the end of all of the objects without encountering a stationary object in a non-unique position, then we
                # can count this satisfaction as long as at least one object was stationary
                if all_unique and encountered_stationary:
                    count += 1

            return count

        elif rule == "count_same_positions":
            preference_name, object_types = self._extract_name_and_types(scoring_expression)
            satisfactions = self._filter_satisfactions(preference_name, object_types, external_mapping)

            # Maps from a set of objects and their stationary positions to the number of satisfactions that use that mapping
            # E.g. if we have two objects, A and B, and A is stationary at position (0, 0) and B is stationary at position (1, 1)
            # for two satisfactions, then we would have the mapping {((A, (0, 0)), (B, (1, 1))): 2}
            stationary_position_counts = defaultdict(int)

            for satisfaction in satisfactions:
                stationary_position_key = []
                for obj in satisfaction.mapping.values():
                    if not any([satisfaction.start < time < satisfaction.end for time, pos in self.object_movements[obj]]):
                        last_position = max(filter(lambda move: move.time < satisfaction.start, self.object_movements[obj]),
                                            key=lambda move: move.time).pos

                        stationary_position_key.append((obj, tuple(last_position)))

                # If the mapping has at least one stationary object, then increment the appropriate count
                if stationary_position_key != []:
                    stationary_position_counts[tuple(stationary_position_key)] += 1

            # We return the maximal count if there are any satisfactions that have at least one stationary object, otherwise we return 0
            return max(stationary_position_counts.values()) if stationary_position_counts else 0

        # Count the number of satisfactions of the given preference that use distinct variable mappings for the externally
        # quantified variables
        elif rule == "count_once_per_external_objects":
            preference_name, object_types = self._extract_name_and_types(scoring_expression)

            satisfactions = self._filter_satisfactions(preference_name, object_types, external_mapping)
            external_variables = self.preference_handlers[preference_name].additional_variable_mapping.keys()

            count = 0

            keyfunc = lambda satisfaction: "_".join([satisfaction.mapping[var] for var in external_variables])
            for key, group in itertools.groupby(sorted(satisfactions, key=keyfunc), keyfunc):
                count += 1

            return count

        else:
            raise ValueError(f"Error: Unknown rule '{rule}' in scoring expression")

        return 0.0
