from collections import defaultdict
import copy
from enum import Enum
import typing

import numpy as np
import tatsu
import tatsu.ast
import tatsu.infos
from tatsu.infos import ParseInfo
import tqdm

import ast_printer
from ast_parser import ASTParser, ASTNodeInfo, ASTParentMapping, ContextDict
from ast_utils import replace_child
from ast_counter_sampler import RegrowthSampler, ASTSampler, SamplingException, simplified_context_deepcopy
from ast_context_fixer import ASTContextFixer





# class CrossoverSampler(RegrowthSampler):
#     context_fixer: ASTContextFixer
#     node_infos_by_crossover_type_key: typing.Dict[str, typing.List[ASTNodeInfo]]
#     # node_keys_by_id: typing.Dict[str, typing.List[ParseInfo]]
#     # parent_mapping_by_id: typing.Dict[str, ASTParentMapping]
#     population: typing.List[typing.Union[tatsu.ast.AST, tuple]]

#     def __init__(self, crossover_type: CrossoverType,
#         population: typing.List[typing.Union[tatsu.ast.AST, tuple]],
#         sampler: ASTSampler, seed: int = 0, use_tqdm: bool = False):
#         super().__init__(sampler, seed)
#         # a slightly ugly call to get the variable sampler with the right context
#         self.context_fixer = ASTContextFixer(self.sampler.rules['variable_type_def']['var_names']['samplers']['variable'], self.sampler.local_context_propagating_rules, self.rng)
#         self.crossover_type = crossover_type
#         self.population = population
#         self.node_infos_by_crossover_type_key = defaultdict(list)

#         pop_iter = self.population
#         if use_tqdm:
#             pop_iter = tqdm.tqdm(pop_iter)

#         for ast in pop_iter:
#             self.set_source_ast(ast)
#             # self.node_keys_by_id[self.original_game_id] = self.node_keys[:]
#             # self.parent_mapping_by_id[self.original_game_id] = self.parent_mapping.copy()

#             for node_info in self.parent_mapping.values():
#                 self.node_infos_by_crossover_type_key[node_info_to_key(self.crossover_type, node_info)].append(node_info)

#         self.source_ast = None  # type: ignore

#     def sample(self, sample_index: int, external_global_context: typing.Optional[ContextDict] = None,
#         external_local_context: typing.Optional[ContextDict] = None, update_game_id: bool = True,
#         crossover_key_to_use: typing.Optional[str] = None) -> typing.Union[tatsu.ast.AST, tuple]:

#         crossover_node_copy = None
#         crossover_node_copy_fixed = False

#         while not crossover_node_copy_fixed:
#             try:
#                 node_crossover_key = None
#                 node_info = None
#                 if crossover_key_to_use is not None:
#                     while node_crossover_key != crossover_key_to_use:
#                         node_info = self._sample_node_to_update()
#                         node, parent, selector, global_context, local_context = node_info
#                         node_crossover_key = node_info_to_key(self.crossover_type, node_info)

#                 else:
#                     node_info = self._sample_node_to_update()
#                     node_crossover_key = node_info_to_key(self.crossover_type, node_info)

#                 node, parent, selector, global_context, local_context = node_info  # type: ignore
#                 node_infos = self.node_infos_by_crossover_type_key[node_crossover_key]  # type: ignore

#                 if len(node_infos) == 0:
#                     raise SamplingException(f'No nodes found with key {node_crossover_key}')

#                 if len(node_infos) == 1:
#                     crossover_node_info = node_infos[0]
#                     if crossover_node_info is node_info:
#                         raise SamplingException(f'No other nodes found with key {node_crossover_key}')

#                     crossover_node_copy = copy.deepcopy(crossover_node_info[0])
#                     self.context_fixer(crossover_node_copy, global_context=global_context, local_context=local_context)
#                     crossover_node_copy_fixed = True

#                 else:
#                     crossover_node_info = node_info

#                     while crossover_node_info is node_info:
#                         index = self.rng.choice((len(node_infos)))
#                         crossover_node_info = node_infos[index]
#                         crossover_node_copy = copy.deepcopy(crossover_node_info[0])
#                         self.context_fixer(crossover_node_copy, global_context=global_context, local_context=local_context)
#                         crossover_node_copy_fixed = True

#             except SamplingException:
#                 continue

#         # TODO: if the node we're replacing is variable list, do we go to children of the siblings and update them?

#         new_source = copy.deepcopy(self.source_ast)
#         new_parent = self.searcher(new_source, parseinfo=parent.parseinfo)  # type: ignore
#         replace_child(new_parent, selector, crossover_node_copy)  # type: ignore

#         return new_source




# if __name__ == '__main__':
#     import argparse
#     import os
#     from ast_counter_sampler import *
#     from ast_utils import cached_load_and_parse_games_from_file

#     DEFAULT_ARGS = argparse.Namespace(
#         grammar_file=DEFAULT_GRAMMAR_FILE,
#         parse_counter=False,
#         counter_output_path=DEFAULT_COUNTER_OUTPUT_PATH,
#         random_seed=33,
#     )

#     grammar = open(DEFAULT_ARGS.grammar_file).read()
#     grammar_parser = tatsu.compile(grammar)
#     counter = parse_or_load_counter(DEFAULT_ARGS, grammar_parser)
#     sampler = ASTSampler(grammar_parser, counter, seed=DEFAULT_ARGS.random_seed)
#     asts = [ast for ast in cached_load_and_parse_games_from_file('./dsl/interactive-beta.pddl',
#         grammar_parser, False)]


#     crossover_sampler = CrossoverSampler(
#         CrossoverType.SAME_RULE,
#         asts[1:],
#         sampler,
#         DEFAULT_ARGS.random_seed,
#         use_tqdm=True,
#     )
