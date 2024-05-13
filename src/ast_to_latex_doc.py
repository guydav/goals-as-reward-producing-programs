import argparse
from collections import defaultdict, namedtuple
from itertools import chain
import tatsu
import tatsu.ast
import typing
import shutil
import os
import re

import ast_printer
from ast_parser import ASTParser, ASTParentMapper
from ast_utils import cached_load_and_parse_games_from_file

parser = argparse.ArgumentParser()
DEFAULT_GRAMMAR_FILE = './dsl/dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
DEFAULT_TEST_FILES = (
    # '../dsl/problems-few-objects.pddl',
    # '../dsl/problems-medium-objects.pddl',
    # '../dsl/problems-many-objects.pddl',
    './dsl/interactive-beta.pddl',
)
parser.add_argument('-t', '--test-files', action='append', default=[])
DEFAULT_OUTPUT_FILE = './latex/dsl_doc.tex'
parser.add_argument('-o', '--output-file', default=DEFAULT_OUTPUT_FILE)
parser.add_argument('-p', '--print-dsls', action='store_true')
DEFAULT_TEMPLATE_FILE = './latex/template.tex'
parser.add_argument('-f', '--template-file', default=DEFAULT_TEMPLATE_FILE)
parser.add_argument('-c', '--compile-pdf', action='store_true')
parser.add_argument('-n', '--new-data-start')
parser.add_argument('-e', '--external', action='store_true')
parser.add_argument('-u', '--omit-unused-rules', action='store_true')


DEFAULT_PREFIX = r"""\section{DSL Grammar Definitions}

A game is defined by a name, and is expected to be valid in a particular domain, also referenced by a name.
A game is defined by four elements, two of them mandatory, and two optional.
The mandatory ones are the \dsl{constraints} section, which defines gameplay preferences, and the \dsl{scoring} section, which defines how gameplay preferences are counted to arrive at a score for the player in the game.
The optional ones are the \dsl{setup} section, which defines how the environment must be prepared before gameplay can begin, and the \dsl{terminal} conditions, which specify when and how the game ends.

\begin{grammar}
<game> ::= (define (game <ID>) \\
  (:domain <ID>) \\
  (:setup <setup>) \\
  (:constraints <constraints>) \\
  (:terminal <terminal>) \\
  (:scoring <scoring>) \\)

<id> ::= /[a-z0-9][a-z0-9\-]+/  "#" a letter or digit, followed by one or more letters, digits, or dashes
\end{grammar}

We will now proceed to introduce and define the syntax for each of these sections, followed by the non-grammar elements of our domain: predicates, functions, and types.
Finally, we provide a mapping between some aspects of our gameplay preference specification and linear temporal logic (LTL) operators.
"""

TEMPLATE_PLACEHOLDER = '{{BODY}}'
class DSLToLatexParser(ASTParser):
    def __init__(self, template_path, output_path, new_data_start=None, prefix=DEFAULT_PREFIX, postfix_lines=None) -> None:
        super().__init__()
        self.template_path = template_path
        self.output_path = output_path
        self.new_data_start = new_data_start
        self.data_is_new = False

        self.rules_by_section = defaultdict(lambda: defaultdict(list))
        self.processors = []
        self.prefix = prefix if prefix is not None else ''
        self.postfix_lines = postfix_lines if postfix_lines is not None else []

    def register_processor(self, processor):
        self.processors.append(processor)

    def __call__(self, ast, **kwargs):
        if isinstance(ast, tuple) and ast[0] == '(define' and \
            self.new_data_start is not None and ast[1].game_name == self.new_data_start:

            self.data_is_new = True

        return super().__call__(ast, **kwargs)

    def _handle_tuple(self, ast, **kwargs):
        if len(ast) > 0 and isinstance(ast[0], str) and ast[0].startswith('(:'):
            kwargs['section'] = ast[0][2:]

        return super()._handle_tuple(ast, **kwargs)

    def _handle_ast(self, ast, **kwargs):
        if 'section' in kwargs and ast.parseinfo:
            self.rules_by_section[kwargs['section']][ast.parseinfo.rule].append((ast, self.data_is_new))

        super()._handle_ast(ast, **kwargs)

    def process(self):
        for processor in self.processors:
            if isinstance(processor, SectionTranslator):
                if processor.section_key in self.rules_by_section:
                    remaining_keys = processor.process(self.rules_by_section[processor.section_key])
                    if remaining_keys:
                        print(f'After processing section {processor.section_key}, the following keys remain: {remaining_keys}')

                    if processor.unused_rules:
                        print(f'After processing section {processor.section_key}, these rules are unused: {processor.unused_rules}')

                else:
                    raise ValueError(f'Found section processor for unknown section key: {processor.section_key}')

            elif isinstance(processor, RuleTypeTranslator):
                for section_data in self.rules_by_section.values():
                    processor.process(section_data)

            elif isinstance(processor, FakeTranslator):
                continue

            else:
                raise ValueError(f'Found processor of unknown type: {type(processor)}')

    def output(self):
        lines = [self.prefix]

        for processor in self.processors:
            lines.extend(processor.output())
            lines.append('\n')

        lines.extend(self.postfix_lines)

        template_text = open(self.template_path).read()
        template_text = template_text.replace(TEMPLATE_PLACEHOLDER, '\n'.join(lines))

        open(self.output_path, 'w').write(template_text)


UNUSED_RULE_OR_ELEMENT_COLOR = 'gray'
NEW_RULE_OR_ELEMENT_COLOR = 'teal'
UNDESCRIBED_ELEMENT_COLOR = 'red'

SETUP_SECTION_KEY = 'setup'
PREFERENCES_SECTION_KEY = 'constraints'
TERMINAL_SECTION_KEY = 'terminal'
SCORING_SECTION_KEY = 'scoring'
PREDICATES_SECTION_KEY = 'predicates'
FUNCTIONS_SECTION_KEY = 'functions'
TYPES_SECTION_KEY = 'types'

PRE_NOTES_KEY = 'pre'
POST_NOTES_KEY = 'post'

NOTES = {
    SETUP_SECTION_KEY: {
        PRE_NOTES_KEY: r"""The setup section specifies how the environment must be transformed from its deterministic initial conditions to a state gameplay can begin at.
Currently, a particular environment room always appears in the same initial conditions, in terms of which objects exist and where they are placed.
Participants in our experiment could, but did not have to, specify how the room must be setup so that their game could be played.

The initial \dsl{setup} element can expand to conjunctions, disjunctions, negations, or quantifications of itself, and then to the \dsl{setup-statement} rule.
\dsl{setup-statement} elements specify two different types of setup conditions: either those that must be conserved through gameplay (`game-conserved'), or those that are optional through gameplay (`game-optional').
These different conditions arise as some setup elements must be maintain through gameplay (for example, a participant specified to place a bin on the bed to throw balls into, it shouldn't move unless specified otherwise), while other setup elements can or must change (if a participant specified to set the balls on the desk to throw them, an agent will have to pick them up (and off the desk) in order to throw them).

Inside the \dsl{setup-statement} tags we find \dsl{super-predicate} elements, which are logical operations and quantifications over other \dsl{super-predicate} elements,  function comparisons (\dsl{function-comparison}, which like predicates also resolve to a truth value), and predicates (\dsl{predicate}).
Function comparisons usually consist of a comparison operator and two arguments, which can either be the evaluation of a function or a number.
The one exception is the case where the comparison operator is the equality operator (=), in which case any number of arguments can be provided.
Finally, the \dsl{predicate} element expands to a predicate acting on one or more objects or variables.
For a full list of the predicates we found ourselves using so far, see \fullref{sec:predicates}.
        """,
    },
    PREFERENCES_SECTION_KEY: {
        PRE_NOTES_KEY: r"""The gameplay preferences specify the core of a game's semantics, capturing how a game should be played by specifying temporal constraints over predicates.
The name for the overall element, \dsl{constraints}, is inherited from the PDDL element with the same name.

The \dsl{constraints} elements expands into one or more preference definitions, which are defined using the \dsl{pref-def} element.
A \dsl{pref-def} either expands to a single preference (\dsl{preference}), or to a \dsl{pref-forall} element, which specifies variants of the same preference for different objects, which can be treated differently in the scoring section.
A \dsl{preference} is defined by a name and a \dsl{preference-quantifier}, which expands to an optional quantification (exists, forall, or neither), inside of which we find the \dsl{preference-body}.

A \dsl{preference-body} expands into one of two options:
The first is a set of conditions that should be true at the end of gameplay, using the \dsl{at-end} operator.
Inside an \dsl{at-end} we find a \dsl{super-predicate}, which like in the setup section, expands to logical operations or quantifications over other \dsl{super-predicate} elements, function comparisons, or predicates.

The second option is specified using the \dsl{then} syntax, which defines a series of temporal conditions that should hold over a sequence of states.
Under a \dsl{then} operator, we find two or more sequence functions (\dsl{seq-func}), which define the specific conditions that must hold and how many states we expect them to hold for.
We assume that there are no unaccounted states between the states accounted for by the different operators -- in other words, the \dsl{then} operators expects to find a sequence of contiguous states that satisfy the different sequence functions.
The operators under a \dsl{then} operator map onto linear temporal logic (LTL) operators, see \fullref{sec:LTL} for the mapping and examples.

The \dsl{once} operator specifies a predicate that must hold for a single world state.
If a \dsl{once} operators appears as the first operator of a \dsl{then} definition, and a sequence of states $S_a, S_{a+1}, \cdots, S_b$ satisfy the \dsl{then} operator, it could be the case that the predicate is satisfied before this sequence of states (e.g. by $S_{a-1}, S_{a-2}$, and so forth).
However, only the final such state, $S_a$, is required for the preference to be satisfied.
The same could be true at the end of the sequence: if a \dsl{then} operator ends with a \dsl{once} term, there could be other states after the final state ($S_{b+1}, S_{b+2}$, etc.) that satisfy the predicate in the \dsl{once} operator, but only one is required.
The \dsl{once-measure} operator is a slight variation of the \dsl{once} operator, which in addition to a predicate, takes in a function evaluation, and measures the value of the function evaluated at the state that satisfies the preference.
This function value can then be used in the scoring definition, see \fullref{sec:scoring}.

A second type of operator that exists is the \dsl{hold} operator.
It specifies that a predicate must hold true in every state between the one in which the previous operator is satisfied, and until one in which the next operator is satisfied.
If a \dsl{hold} operator appears at the beginning or an end of a \dsl{then} sequence, it can be satisfied by a single state,
Otherwise, it must be satisfied until the next operator is satisfied.
For example, in the minimal definition below:
\begin{lstlisting}
(then
    (once (pred_a))
    (hold (pred_b))
    (once (pred_c))
)
\end{lstlisting}
To find a sequence of states $S_a, S_{a+1}, \cdots, S_b$ that satisfy this \dsl{then} operator, the following conditions must hold true: (1) pred_a is true at state $S_a$, (2) pred_b is true in all states $S_{a+1}, S_{a+2}, \cdots, S_{b-2}, S_{b-1}$, and (3) pred_c is true in state $S_b$.
There is no minimal number of states that the hold predicate must hold for.

The last operator is \dsl{hold-while}, which offers a variation of the \dsl{hold} operator.
A \dsl{hold-while} receives at least two predicates.
The first acts the same as predicate in a \dsl{hold} operator.
The second (and third, and any subsequent ones), must hold true for at least state while the first predicate holds, and must occur in the order specified.
In the example above, if we substitute \lstinline{(hold (pred_b))} for \lstinline{(hold-while (pred_b) (pred_d) (pred_e))}, we now expect that in addition to ped_b being true in all states $S_{a+1}, S_{a+2}, \cdots, S_{b-2}, S_{b-1}$, that there is some state $S_d, d \in [a+1, b-1]$ where pred_d holds, and another state, $S_e, e \in [d+1, b-1]$ where pred_e holds.
        """,  # .format(unused_color=UNUSED_RULE_OR_ELEMENT_COLOR) # Any syntax elements that are defined (because at some point a game needed them) but are currently unused (in the interactive games) will appear in {{ \color{{{unused_color}}} {unused_color} }}.
        POST_NOTES_KEY: r"""For the full specification of the \dsl{super-predicate} element, see \fullref{sec:setup} above.
        """,
    },
    TERMINAL_SECTION_KEY: {
        PRE_NOTES_KEY: r"""Specifying explicit terminal conditions is optional, and while some of our participants chose to do so, many did not.
Conditions explicitly specified in this section terminate the game.
If none are specified, a game is assumed to terminate whenever the player chooses to end the game.

The terminal conditions expand from the \dsl{terminal} element, which can expand to logical conditions on nested \dsl{terminal} elements, or to a terminal comparison.
The terminal comparison (\dsl{terminal-comp}) expands to one of three different types of copmarisons: \dsl{terminal_time_comp}, a comparison between the total time spent in the game (\lstinline{(total-time)}) and a time number token, \dsl{terminal-score-comp}, a comparison between the total score (\lstinline{(total-score)}) and a score number token, or \dsl{terminal-pref-count-comp}, a comparison between a scoring expression (\dsl{scoring-expr}, see below) and a preference count number token.
In most cases, the scoring expression is a preference counting operation.
        """,
        POST_NOTES_KEY: r"""For the full specification of the \dsl{scoring-expr} element, see \fullref{sec:scoring} below.
        """,
    },
    SCORING_SECTION_KEY: {
        PRE_NOTES_KEY: r"""Scoring rules specify how to count preferences (count once, once for each unique objects that fulfill the preference, each time a preference is satisfied, etc.), and the arithmetic to combine preference counts to a final score in the game.

A \dsl{scoring-expr} can be defined by arithmetic operations on other scoring expressions, references to the total time or total score (for instance, to provide a bonus if a certain score is reached), comparisons between scoring expressions (\dsl{scoring-comp}), or by preference evaluation rules.
Various preference evaluation modes can expand the \dsl{preference-eval} rule, see the full list and descriptions below.
        """  # .format(unused_color=UNUSED_RULE_OR_ELEMENT_COLOR) # Any syntax elements that are defined (because at some point a game needed them) but are currently unused (in the interactive games) will appear in {{ \color{{{unused_color}}} {unused_color} }}.
    },
    PREDICATES_SECTION_KEY: {
        PRE_NOTES_KEY: r"""The following section describes the predicates we define.
        Predicates operate over a specified number of arguments, which can be variables or object names, and return a boolean value (true/false).
"""  # .format(undescribed_color=UNDESCRIBED_ELEMENT_COLOR) # Any predicates I forgot to provide a description for will appear in {{ \color{{{undescribed_color}}} {undescribed_color} }}.
    },
    FUNCTIONS_SECTION_KEY: {
        PRE_NOTES_KEY: r"""he following section describes the functions we define.
        Functions operate over a specified number of arguments, which can be variables or object names, and return a number.""",
    },
    TYPES_SECTION_KEY: {
        PRE_NOTES_KEY: r"""The types are currently not defined as part of the grammar, other than the small list of \dsl{object-name} tokens that can be directly referred to, and are marked with an asterisk below.
        The following enumerates all expansions of the various \dsl{type} rules:
        """  # .format(undescribed_color=UNDESCRIBED_ELEMENT_COLOR) # Any types we forgot to provide a description for will appear in {{\color{{{undescribed_color}}}{undescribed_color} }}.
    }
}

class DSLTranslator:
    def __init__(self, section_key, section_type, section_name=None, external_mode=False, omit_unused=False, notes=NOTES):
        self.section_key = section_key
        self.section_type = section_type
        self.section_name = section_name if section_name is not None else section_key.capitalize()

        self.external_mode = external_mode
        self.omit_unused = omit_unused

        self.notes = notes

        self.lines = []

    def process(self, section_data):
        raise NotImplementedError()

    def output(self):
        section_notes = self.notes[self.section_key] if self.section_key in self.notes else {}
        return [
            f'\\subsection{{{self.section_name}}} \\label{{sec:{self.section_key.lower().replace(" ", "-")}}}',
            section_notes[PRE_NOTES_KEY] if PRE_NOTES_KEY in section_notes else '',
            f'\\begin{{{self.section_type}}}'
        ] + self.lines + [
            f'\\end{{{self.section_type}}}',
            section_notes[POST_NOTES_KEY] if POST_NOTES_KEY in section_notes else '',
        ]


class FakeTranslator:
    def __init__(self, output_data):
        self.output_data = output_data

    def process(self, section_data):
        pass

    def output(self):
        if isinstance(self.output_data, list):
            return self.output_data

        return [self.output_data]

class SectionTranslator(DSLTranslator):
    def __init__(self, section_key, core_blocks, additional_blocks=None, section_name=None, consider_used_rules=None,
                 section_type='grammar', unused_rule_color=UNUSED_RULE_OR_ELEMENT_COLOR, new_rule_color=NEW_RULE_OR_ELEMENT_COLOR,
                 external_mode=False, omit_unused=False, consider_described_rules: typing.Optional[typing.Sequence[typing.Union[str, re.Pattern]]] = None):
        super().__init__(section_key, section_type, section_name, external_mode=external_mode, omit_unused=omit_unused)
        self.section_key = section_key
        self.core_blocks = core_blocks
        self.additional_blocks = additional_blocks if additional_blocks is not None else []
        self.consider_used_rules = consider_used_rules if consider_used_rules is not None else []
        self.unused_rule_color = unused_rule_color
        self.new_rule_color = new_rule_color
        self.consider_described_rules = consider_described_rules

        self.lines = []
        self.unused_rules = []

    def process(self, section_data):
        keys = set(section_data.keys())

        for blocks, is_core in zip((self.core_blocks, self.additional_blocks), (True, False)):
            for block_text, block_rules in blocks:

                if isinstance(block_rules, str):
                    if block_rules not in keys and block_rules not in self.consider_used_rules:
                        self.unused_rules.append(block_rules)

                        if not is_core or self.omit_unused:
                            continue

                        if not self.external_mode:
                            block_text = f'{{ \\color{{{self.unused_rule_color}}} {block_text} }}'

                    elif block_rules in keys:
                        _, block_is_new = zip(*section_data[block_rules])
                        rule_is_new = all(block_is_new)
                        if rule_is_new and not self.external_mode:
                            block_text = f'{{ \\color{{{self.new_rule_color}}} {block_text} }}'

                        keys.remove(block_rules)

                    self.lines.append(block_text)

                elif is_core or any([rule in keys for rule in block_rules]):
                    self.lines.extend(block_text.split('\n'))
                    for rule in block_rules:
                        if isinstance(rule, re.Pattern):
                            keys -= set([key for key in keys if rule.match(key)])

                        elif rule in keys:
                            keys.remove(rule)

                        elif is_core and rule not in self.consider_used_rules:
                            self.unused_rules.append(rule)

                self.lines.append('')

            self.lines.append('')

        if self.consider_described_rules is not None:
            for rule in self.consider_described_rules:
                if isinstance(rule, str):
                    if rule in keys:
                        keys.remove(rule)

                elif isinstance(rule, re.Pattern):
                    keys -= set([key for key in keys if rule.match(key)])

        return keys

DEFAULT_RULE_SECTION_TYPE = 'lstlisting'

class RuleTypeTranslator(DSLTranslator):
    def __init__(self, section_key, rule_to_value_extractors, output_lines_func, section_type=DEFAULT_RULE_SECTION_TYPE,
        descriptions=None, section_name=None, external_mode=False, omit_unused=False):

        super().__init__(section_key, section_type, section_name, external_mode=external_mode, omit_unused=omit_unused)
        self.rule_to_value_extractors = rule_to_value_extractors
        self.output_lines_func = output_lines_func
        self.count_data = defaultdict(lambda: 0)
        self.is_new_data = defaultdict(lambda: True)
        self.additional_data = defaultdict(lambda: defaultdict(list))
        self.descriptions = descriptions if descriptions is not None else {}

    def process(self, section_data):
        for rule, (key_extractor, additional_extractors) in self.rule_to_value_extractors.items():
            if rule in section_data:
                for instance, instance_is_new in section_data[rule]:
                    instance_key = instance[key_extractor] if isinstance(key_extractor, str) else key_extractor(instance)

                    if instance_key:
                        if isinstance(instance_key, str):
                            self._handle_single_key(instance, instance_key, instance_is_new, additional_extractors)

                        else:
                            for key in instance_key:
                                self._handle_single_key(instance, key, instance_is_new, additional_extractors)

    def _handle_single_key(self, instance, instance_key, instance_is_new, additional_extractors):
        self.count_data[instance_key] += 1
        self.is_new_data[instance_key] = self.is_new_data[instance_key] and instance_is_new

        if additional_extractors:
            for additional_extractor in additional_extractors:
                key, value = additional_extractor(instance, instance_key)
                if key is not None:
                    self.additional_data[instance_key][key].append(value)

    def output(self):
        self.lines = self.output_lines_func(self.count_data, self.is_new_data, self.additional_data, self.descriptions, self.external_mode)
        return super().output()


def _format_line_to_color(line, color):
    return f'(*\\color{{{color}}} {line}*)'


def predicate_data_to_lines(count_data, is_new_data, additional_data, descriptions, external_mode=False, omit_unused=False):
    lines = []

    for key in sorted(count_data.keys()):
        count = count_data[key]
        n_args = list(sorted(set(additional_data[key]['n_args'])))
        arg_str = ' '.join([f'<arg{i + 1}>' for i in range(n_args[0])]) if len(n_args) == 1 else f'<{" or ".join([str(n) for n in n_args])} arguments>'

        line = f'({key} {arg_str}) [{count} reference{"s" if count != 1 else ""}] {"; " + descriptions[key] if key in descriptions else ""}'
        if key not in descriptions and not external_mode:
            line = _format_line_to_color(line, UNDESCRIBED_ELEMENT_COLOR)

        elif is_new_data[key] and not external_mode:
            line = _format_line_to_color(line, NEW_RULE_OR_ELEMENT_COLOR)

        lines.append(line)

    return lines


TypeDesc = namedtuple('TypeDesc', ('key', 'description', 'preformatted', 'can_directly_refer'), defaults=(None, '', False, False))


def section_separator_typedesc(section_name):
    return TypeDesc(section_name, f'---------- (* \\textbf{{{section_name}}} *) ----------', True)


def type_data_to_lines(count_data, is_new_data, additional_data, descriptions, external_mode=True, omit_unused=False):
    lines = []
    unused_keys = set(count_data.keys())

    directly_refer_mark = r'*'

    for type_desc in descriptions:
        key, description, preformatted, can_directly_refer = type_desc

        if preformatted:
            lines.append(description)

        else:
            if key in unused_keys:
                unused_keys.remove(key)

            count = count_data[key] if key in count_data else 0
            # TODO: consider doing something with co-ocurrence data
            line = f'{key}{directly_refer_mark if can_directly_refer else ""} [{count if count != 0 else "0"} reference{"s" if count != 1 else ""}] {"; " + description if description else ""}'

            # if count == 0:
            #     if omit_unused:
            #         continue

            #     if not external_mode:
            #         line = _format_line_to_color(line, UNUSED_RULE_OR_ELEMENT_COLOR)

            # elif is_new_data[key] and not external_mode:
            #     line = _format_line_to_color(line, NEW_RULE_OR_ELEMENT_COLOR)

            lines.append(line)

    if unused_keys and not external_mode:
        lines.append(section_separator_typedesc('Undescribed types').description)
        for key in unused_keys:
            count = count_data[key]
            line = _format_line_to_color(f'{key} [{count} reference{"s" if count != 1 else ""}]', UNDESCRIBED_ELEMENT_COLOR)
            lines.append(line)

    return lines


# def _type_name_sorting(type_name):
#     if type_name == 'game_object': return -1
#     if type_name == 'block': return 0
#     if '_block' in type_name: return 1
#     if type_name == 'ball': return 2
#     if 'ball' in type_name: return 3
#     return ord(type_name[0])


FUNCTION_COMPARISON = 'function_comparison'
VARIABLE_LIST = 'variable_list'
PREDICATE = 'predicate'
FUNCTION = 'function_eval'
SHARED_BLOCKS = {
    FUNCTION_COMPARISON: (r"""<function-comparison> ::= "#" A function comparison: either comparing two function evaluations, or checking that two ore more functions evaluate to the same result.
    \alt (<comp-op> <function-eval-or-number> <function-eval-or-number>)
    \alt (= <function-eval-or-number>$^+$)

<comp-op> ::=  \textlangle \ | \textlangle = \ | = \ | \textrangle \ | \textrangle = "#" Any of the comparison operators.

<function-eval-or-number> ::= <function-eval> | <comparison-arg-number>

<comparison-arg-number> ::= <number>

<number> ::=  /-?\textbackslash d*\textbackslash .?\textbackslash d+/  "#" A number, either an integer or a float.

<function-eval> ::= "#" See valid expansions in a separate section below
""", ('function_comparison', 'multiple_args_equal_comparison', 'two_arg_comparison', 'comparison_arg', FUNCTION, 'comparison_arg_number_value', re.compile('function_[A-Za-z_]+'))),

    VARIABLE_LIST: (r"""<variable-list> ::= (<variable-def>$^+$) "#" One or more variables definitions, enclosed by parentheses.

<variable-def> ::= <variable-type-def> | <color-variable-type-def> | <orientation-variable-type-def> | <side-variable-type-def> "#" Colors, sides, and orientations are special types as they are not interchangable with objects.

<variable-type-def> ::= <variable>$^+$ - <type-def> "#" Each variable is defined by a variable (see next) and a type (see after).

<color-variable-type-def> ::= <color-variable>$^+$ - <color-type-def> "#" A color variable is defined by a variable (see below) and a color type.

<orientation-variable-type-def> ::= <orientation-variable>$^+$ - <orientation-type-def> "#" An orientation variable is defined by a variable (see below) and an orientation type.

<side-variable-type-def> ::= <side-variable>$^+$ - <side-type-def> "#" A side variable is defined by a variable (see below) and a side type.

<variable> ::= /\textbackslash?[a-w][a-z0-9]*/  "#" a question mark followed by a lowercase a-w, optionally followed by additional letters or numbers.

<color-variable> ::= /\textbackslash?x[0-9]*/  "#" a question mark followed by an x and an optional number.

<orientation-variable> ::= /\textbackslash?y[0-9]*/  "#" a question mark followed by an y and an optional number.

<side-variable> ::= /\textbackslash?z[0-9]*/  "#" a question mark followed by an z and an optional number.

<type-def> ::= <object-type> | <either-types> "#" A veriable type can either be a single name, or a list of type names, as specified below

<color-type-def> ::= <color-type> | <either-color-types> "#" A color variable type can either be a single color name, or a list of color names, as specified below

<orientation-type-def> ::= <orientation-type> | <either-orientation-types> "#" An orientation variable type can either be a single orientation name, or a list of orientation names, as specified below

<side-type-def> ::= <side-type> | <either-side-types> "#" A side variable type can either be a single side name, or a list of side names, as specified below

<either-types> ::= (either <object-type>$^+$)

<either-color-types> ::= (either <color>$^+$)

<either-orientation-types> ::= (either <orientation>$^+$)

<either-side-types> ::= (either <side>$^+$)

<object-type> ::= <name>

<name> ::= /[A-Za-z][A-za-z0-9\_]+/  "#" a letter, followed by one or more letters, digits, or underscores

<color-type> ::= 'color'

<color> ::= 'blue' | 'brown' | 'gray' | 'green' | 'orange' | 'pink' | 'purple' | 'red' | 'tan' | 'white' | 'yellow'

<orientation-type> ::= 'orientation'

<orientation> ::= 'diagonal' | 'sideways' | 'upright' | 'upside_down'

<side-type> ::= 'side'

<side> ::= 'back' | 'front' | 'left' | 'right'""", ('variable_list', 'variable_type_def', 'color_variable_type_def', 'orientation_variable_type_def',  'side_variable_type_def',
                                                    'type_definition', 'color_type_definition', 'orientation_type_definition', 'side_type_definition',
                                                    'either_types', 'either_color_types', 'either_orientation_types', 'either_side_types',
                                                    'object_type', 'color_type', 'color', 'orientation_type', 'orientation', 'side_type', 'side', 'name',
                                                    )),

    PREDICATE: (r"""<predicate> ::= "#" See valid expansions in a separate section below

<predicate-or-function-term> ::= <object-name> | <variable> "#" A predicate or function term can either be an object name (from a small list allowed to be directly referred to) or a variable.

<predicate-or-function-color-term> ::= <color> | <color-variable>

<predicate-or-function-orientation-term> ::= <orientation> | <orientation-variable>

<predicate-or-function-side-term> ::= <side> | <side-variable>

<predicate-or-function-type-term> ::= <object-type> | <variable>

<object_name> ::= 'agent' | 'bed' | 'desk' | 'door' | 'floor' | 'main_light_switch' | 'mirror' | 'room_center' | 'rug' | 'side_table' | 'bottom_drawer' | 'bottom_shelf' | 'east_sliding_door' | 'east_wall' | 'north_wall' | 'south_wall' | 'top_drawer' | 'top_shelf' | 'west_sliding_door' | 'west_wall'
""",
('predicate', 'predicate_or_function_term', 'predicate_or_function_color_term', 'predicate_or_function_orientation_term', 'predicate_or_function_side_term', 'predicate_or_function_type_term', 'object_name', re.compile('predicate_[A-Za-z_]+'))),
}


SETUP_BLOCKS = (
    (r"""<setup> ::= (and <setup> <setup>$^+$) "#" A setup can be expanded to a conjunction, a disjunction, a quantification, or a setup statement (see below).
    \alt (or <setup> <setup>$^+$)
    \alt (not <setup>)
    \alt (exists (<typed list(variable)>) <setup>)
    \alt (forall (<typed list(variable)>) <setup>)
    \alt <setup-statement>""", ('setup', 'setup_and', 'setup_or', 'setup_not', 'setup_exists', 'setup_forall', 'setup_statement')),

    (r"""<setup-statement> ::= "#" A setup statement specifies that a predicate is either optional during gameplay or must be preserved during gameplay.
    \alt (game-conserved <super-predicate>)
    \alt (game-optional <super-predicate>)""", ('setup_game_conserved', 'setup_game_optional',)),

    (r"""<super-predicate> ::= "#" A super-predicate is a conjunction, disjunction, negation, or quantification over another super-predicate. It can also be directly a function comparison or a predicate.
    \alt (and <super-predicate>$^+$)
    \alt (or <super-predicate>$^+$)
    \alt (not <super-predicate>
    \alt (exists (<typed list(variable)>) <super-predicate>)
    \alt (forall (<typed list(variable)>) <super-predicate>)
    \alt <f-comp>
    \alt <predicate>""", ('super_predicate', 'super_predicate_and', 'super_predicate_or', 'super_predicate_not', 'super_predicate_forall', 'super_predicate_exists')),
)


PREFERENCES_BLOCKS = (
    (r"""<constraints> ::= <pref-def> | (and <pref-def>$^+$)  "#" One or more preferences.

<pref-def> ::= <pref-forall> | <preference> "#" A preference definitions expands to either a forall quantification (see below) or to a preference.

<pref-forall> ::= (forall <variable-list> <preference>) "#" this syntax is used to specify variants of the same preference for different objects, which differ in their scoring. These are specified using the <pref-name-and-types> syntax element's optional types, see scoring below.

<preference> ::= (preference <name> <preference-quantifier>) "#" A preference is defined by a name and a quantifer that includes the preference body.

<preference-quantifier> ::= "#" A preference can quantify exsistentially or universally over one or more variables, or none.
\alt (exists (<variable-list>) <preference-body>
\alt  (forall (<variable-list>) <preference-body>)
\alt <preference-body>)

<preference-body> ::=  <then> | <at-end>""", ('preferences', 'preference', 'pref_forall', 'pref_def', 'pref_body', 'pref_body_exists', 'pref_forall_prefs')), # | <always>

    (r'<at-end> ::= (at-end <super-predicate>) "#" Specifies a prediicate that should hold in the terminal state.', 'at_end'),

    # (r'<always> ::= (always <super-predicate>)', 'always'),

    (r"""<then> ::= (then <seq-func> <seq-func>$^+$) "#" Specifies a series of conditions that should hold over a sequence of states -- see below for the specific operators (<seq-func>s), and Section 2 for translation of these definitions to linear temporal logicl (LTL).

<seq-func> ::= <once> | <once-measure> | <hold> | <hold-while> "#" Four of thse temporal sequence functions currently exist: """, ('then', 'seq_func')),  #  | <hold-for> | <hold-to-end> \alt <forall-seq>

    (r'<once> ::= (once <super-predicate>) "#" The predicate specified must hold for a single world state.', 'once'),

    (r'<once-measure> ::= (once <super-predicate> <function-eval>) "#" The predicate specified must hold for a single world state, and record the value of the function evaluation, to be used in scoring.', 'once_measure'),

    (r'<hold> ::= (hold <super-predicate>) "#" The predicate specified must hold for every state between the previous temporal operator and the next one.', 'hold'),

    (r'<hold-while> ::= (hold-while <super-predicate> <super-predicate>$^+$) "#" The first predicate specified must hold for every state between the previous temporal operator and the next one. While it does, at least one state must satisfy each of the predicates specified in the second argument onward'  , 'while_hold'),

    # (r'<hold-for> ::= (hold-for <number> <super-predicate>)', 'hold_for'),

    # (r'<hold-to-end> ::= (hold-to-end <super-predicate>)', 'hold_to_end'),

    # (r'<forall-seq> ::= (forall-sequence (<variable-list>) <then>)', 'forall_seq'),

#     (r"""<super-predicate> ::= (and <super-predicate>$^+$) \alt
#     (or <super-predicate>$^+$) \alt
#     (not <super-predicate>) \alt
#     (exists <variable-list> <super-predicate>) \alt
#     (forall <variable-list> <super-predicate>) \alt
#     <predicate> \alt
#     <f-comp>
# """, ('pref_predicate_and', 'pref_predicate_or', 'pref_predicate_not', 'pref_predicate_exists', 'pref_predicate_forall' ))
)


TERMINAL_BLOCKS = (
    (r"""<terminal> ::= "#" The terminal condition is specified by a conjunction, disjunction, negation, or comparson (see below).
        \alt (and <terminal>$^+$)
        \alt (or <terminal>$+$)
        \alt (not <terminal>)
        \alt <terminal-comp>""", ('terminal', 'terminal_and', 'terminal_or', 'terminal_not')),

    (r"""<terminal-comp> ::= "#" We support three ttypes of terminal comparisons:
        \alt <terminal-time-comp>
        \alt <terminal-score-comp>
        \alt <terminal-pref-count-comp>


    <terminal-time-comp> ::= (<comp-op> (total-time) <time-number>) "#" The total time of the game must satisfy the comparison.

    <terminal-score-comp> ::= (<comp-op> (total-score) <score-number>) "#" The total score of the game must satisfy the comparison.

    <terminal-pref-count-comp> ::= (<comp-op> <scoring-expr> <preference-count-number>) "#" The number of times the preference specified by the name and types must satisfy the comparison.

    <time-number> ::= <number>  "#" Separate type so the we can learn a separate distribution over times than, say, scores.

    <score-number> ::= <number>

    <preference-count-number> ::= <number>

    <comp-op> ::=  \textlangle \ | \textlangle = \ | = \ | \textrangle \ | \textrangle =""", ('terminal_comp', 'terminal_time_comp', 'terminal_score_comp', 'terminal_pref_count_comp', 'total_time', 'total_score',
                                                                                              'time_number_value', 'score_number_value', 'pref_count_number_value')),
)


SCORING_BLOCKS = (
    (r"""<scoring> ::= <scoring-expr> "#" The scoring conditions maximize a scoring expression. """, ('scoring_expr', )),

    (r"""<scoring-expr> ::= "#" A scoring expression can be an arithmetic operation over other scoring expressions, a reference to the total time or score, a comparison, or a preference scoring evaluation.
        \alt <scoring-external-maximize>
        \alt <scoring-external-minimize>
        \alt (<multi-op> <scoring-expr>$^+$) "#" Either addition or multiplication.
        \alt (<binary-op> <scoring-expr> <scoring-expr>) "#" Either division or subtraction.
        \alt (- <scoring-expr>)
        \alt (total-time)
        \alt (total-score)
        \alt <scoring-comp>
        \alt <preference-eval>
        \alt <scoring-number-value>

    """, ('scoring_multi_expr', 'scoring_binary_expr', 'scoring_neg_expr', 'scoring_expr_or_number')),

    (r"""<scoring-external-maximize> ::= (external-forall-maximize <scoring-expr>) "#" For any preferences under this expression inside a (forall ...), score only for the single externally-quantified object that maximizes this scoring expression.
    """, ('scoring_external_maximize',)),

    (r"""<scoring-external-minimize> ::= (external-forall-minimize <scoring-expr>) "#" For any preferences under this expression inside a (forall ...), score only for the single externally-quantified object that minimizes this scoring expression.
    """, ('scoring_external_minimize',)),

    (r"""<scoring-comp> ::=  "#" A scoring comparison: either comparing two expressions, or checking that two ore more expressions are equal.
        \alt (<comp-op> <scoring-expr> <scoring-expr>)
        \alt (= <scoring-expr>$^+$)
    """, ('scoring_comparison', 'scoring_comp', 'scoring_equals_comp')),

    (r"""<preference-eval> ::= "#" A preference evaluation applies one of the scoring operators (see below) to a particular preference referenced by name (with optional types).
        \alt <count>
        \alt <count-overlapping>
        \alt <count-once>
        \alt <count-once-per-objects>
        \alt <count-measure>
        \alt <count-unique-positions>
        \alt <count-same-positions>
        \alt <count-once-per-external-objects>

    """, 'preference_eval'),

    (r'<count> ::= (count <pref-name-and-types>) "#" Count how many times the preference is satisfied by non-overlapping sequences of states.', 'count'),
    (r'<count-overlapping> ::= (count-overlapping <pref-name-and-types>) "#" Count how many times the preference is satisfied by overlapping sequences of states.', 'count_overlapping'),
    (r'<count-once> ::= (count-once <pref-name-and-types>) "#" Count whether or not this preference was satisfied at all.', 'count_once'),
    (r'<count-once-per-objects> ::= (count-once-per-objects <pref-name-and-types>) "#" Count once for each unique combination of objects quantified in the preference that satisfy it.', 'count_once_per_objects'),
    # (r'<count-longest> ::= (count-longest <pref-name-and-types>) "#" count the longest (by number of states) satisfication of this preference', 'count_longest'),
    # (r'<count-shortest> ::= (count-shortest <pref-name-and-types>) "#" count the shortest satisfication of this preference ', 'count_shortest'),
    # (r'<count-total> ::= (count-total <pref-name-and-types>) "#" count how many states in total satisfy this preference', 'count_total'),
    # (r'<count-increasing-measure> ::= (count-increasing-measure <pref-name-and-types>) "#" currently unused, will clarify definition if it surfaces again', 'count_increasing_measure'),
    (r'<count-measure> ::= (count-measure <pref-name-and-types>) "#" Can only be used in preferences including a <once-measure> modal, maps each preference satistifaction to the value of the function evaluation in the <once-measure>.', 'count_measure'),
    (r'<count-unique-positions> ::= (count-unique-positions <pref-name-and-types>) "#" Count how many times the preference was satisfied with quantified objects that remain stationary within each preference satisfcation, and have different positions between different satisfactions.', 'count_unique_positions'),
    (r'<count-same-positions> ::= (count-same-positions <pref-name-and-types>) "#" Count how many times the preference was satisfied with quantified objects that remain stationary within each preference satisfcation, and have (approximately) the same position between different satisfactions.', 'count_same_positions'),
    # (r'<note> : "#" All of the count-maximal-... operators refer to counting only for preferences inside a (forall ...), and count only for the object quantified externally that has the most preference satisfactions to it. If there exist multiple preferences in a single (forall ...) block, score for the single object that satisfies the most over all such preferences.', 'maximal_explainer'),
    # (r'<count-maximal-nonoverlapping> ::= (count-maximal-nonoverlapping <pref-name-and-types>) "#" For the single externally quantified object with the most satisfcations, count non-overlapping satisfactions of this preference.', 'count_maximal_nonoverlapping'),
    # (r'<count-maximal-overlapping> ::= (count-maximal-overlapping <pref-name-and-types>) "#" For the single externally quantified object with the most satisfcations, count how many satisfactions of this preference with different objects overlap in their states.', 'count_maximal_overlapping'),
    # (r'<count-maximal-once-per-objects> ::= (count-maximal-once-per-objects <pref-name-and-types>) "#" For the single externally quantified object with the most satisfcations, count this preference for each set of quantified objects that satisfies it.', 'count_maximal_once_per_objects'),
    # (r'<count-maximal-once> ::= (count-maximal-once <pref-name-and-types>) "#" For the externally quantified object with the most satisfcations (across all preferences in the same (forall ...) block), count this preference at most once.', 'count_maximal_once'),
    (r'<count-once-per-external-objects> ::=  (count-once-per-external-objects <pref-name-and-types>) "#" Similarly to count-once-per-objects, but counting only for each unique object or combination of objects quantified in the (forall ...) block including this preference.',  'count_once_per_external_objects'),

    (r"""<pref-name-and-types> ::= <name> <pref-object-type>$^*$ "#" The optional <pref-object-type>s are used to specify a particular instance of the preference for a given object, see the <pref-forall> syntax above.

    <pref-object-type> ::= : <type-name>  "#" The optional type name specification for the above syntax. For example, pref-name:dodgeball would refer to the preference where the first quantified object is a dodgeball.

    <scoring-number-value> ::= <number>
     """, ('pref_name_and_types', 'pref_object_type', 'scoring_number_value')),
)

PREDICATE_DESCRIPTIONS = {
    'above': 'Is the first object above the second object?',
    'adjacent': 'Are the two objects adjacent? [will probably be implemented as distance below some threshold]',
    'adjacent_side': 'Are the two objects adjacent on the sides specified? Specifying a side for the second object is optional, allowing to specify <obj1> <side1> <obj2> or <obj1> <side1> <obj2> <side2>',
    'agent_crouches': 'Is the agent crouching?',
    'agent_holds': 'Is the agent holding the object?',
    'between': 'Is the second object between the first object and the third object?',
    'broken': 'Is the object broken?',
    'equal_x_position': 'Are these two objects (approximately) in the same x position? (in our environment, x, z are spatial coordinates, y is the height)',
    'equal_z_position': 'Are these two objects (approximately) in the same z position? (in our environment, x, z are spatial coordinates, y is the height)',
    'faces': 'Is the front of the first object facing the front of the second object?',
    'game_over': 'Is this the last state of gameplay?',
    'game_start': 'Is this the first state of gameplay?',
    'in': 'Is the second argument inside the first argument? [a containment check of some sort, for balls in bins, for example]',
    'in_motion': 'Is the object in motion?',
    'is_setup_object': 'Is this the object of the same type referenced in the setup?',
    'near': 'Is the second object near the first object? [implemented as distance below some threshold]',
    'object_orientation': 'Is the first argument, an object, in the orientation specified by the second argument? Used to check if an object is upright or upside down',
    'on': 'Is the second object on the first one?',
    'open': 'Is the object open? Only valid for objects that can be opened, such as drawers.',
    'opposite': 'So far used only with walls, or sides of the room, to specify two walls opposite each other in conjunction with other predicates involving these walls',
    'rug_color_under': 'Is the color of the rug under the object (first argument) the color specified by the second argument?',
    'same_color': 'If two objects, do they have the same color? If one is a color, does the object have that color?',
    'same_object': 'Are these two variables bound to the same object?',
    'same_type': 'Are these two objects of the same type? Or if one is a direct reference to a type, is this object of that type?',
    'toggled_on': 'Is this object toggled on?',
    'touch': 'Are these two objects touching?',
}

FUNCTION_DESCRIPTIONS = {
    'building_size': 'Takes in an argument of type building, and returns how many objects comprise the building (as an integer).',
    'color': 'Take in an argument of type object, and returns the color of the object (as a color type object).',
    'distance': 'Takes in two arguments of type object, and returns the distance between the two objects (as a floating point number).',
    'distance_side': 'Similarly to the adjacent_side predicate, but applied to distance. Takes in three or four arguments, either <obj1> <side1> <obj2> or <obj1> <side1> <obj2> <side2>, and returns the distance between the first object on the side specified to the second object (optionally to its specified side).',
    'x_position': 'Takes in an argument of type object, and returns the x position of the object (as a floating point number).',
}


TYPE_DESCRIPTIONS = (
    TypeDesc('game_object', 'Parent type of all objects'),
    TypeDesc('agent', 'The agent', can_directly_refer=True),
    TypeDesc('building', 'Not a real game object, but rather, a way to refer to structures the agent builds'),
    section_separator_typedesc('Blocks'),
	TypeDesc('block', 'Parent type of all block types:'),
	TypeDesc('bridge_block'),
    TypeDesc('bridge_block_green'),
    TypeDesc('bridge_block_pink'),
    TypeDesc('bridge_block_tan'),
	TypeDesc('cube_block'),
    TypeDesc('cube_block_blue'),
    TypeDesc('cube_block_tan'),
    TypeDesc('cube_block_yellow'),
    TypeDesc('cylindrical_block'),
    TypeDesc('cylindrical_block_blue'),
    TypeDesc('cylindrical_block_green'),
    TypeDesc('cylindrical_block_tan'),
	TypeDesc('flat_block'),
    TypeDesc('flat_block_gray'),
    TypeDesc('flat_block_tan'),
    TypeDesc('flat_block_yellow'),
	TypeDesc('pyramid_block'),
    TypeDesc('pyramid_block_blue'),
    TypeDesc('pyramid_block_red'),
    TypeDesc('pyramid_block_yellow'),
    TypeDesc('tall_cylindrical_block'),
    TypeDesc('tall_cylindrical_block_green'),
    TypeDesc('tall_cylindrical_block_tan'),
    TypeDesc('tall_cylindrical_block_yellow'),
    TypeDesc('tall_rectangular_block'),
    TypeDesc('tall_rectangular_block_blue'),
    TypeDesc('tall_rectangular_block_green'),
    TypeDesc('tall_rectangular_block_tan'),
    TypeDesc('triangle_block'),
    TypeDesc('triangle_block_blue'),
    TypeDesc('triangle_block_green'),
    TypeDesc('triangle_block_tan'),
    section_separator_typedesc('Balls'),
	TypeDesc('ball', 'Parent type of all ball types:'),
	TypeDesc('beachball'),
	TypeDesc('basketball'),
	TypeDesc('dodgeball'),
    TypeDesc('dodgeball_blue'),
    TypeDesc('dodgeball_red'), #, '(* \\textbf Do we want to specify colored objects or not? *)'),
	TypeDesc('dodgeball_pink'), #, '(* \\textbf Do we want to specify colored objects or not? *)'),
	TypeDesc('golfball'),
    TypeDesc('golfball_green'), # '(* \\textbf Do we want to specify colored objects or not? *)'),
    TypeDesc('golfball_white'),
	section_separator_typedesc('Colors'),
    TypeDesc('color', 'Likewise, not a real game object, mostly used to refer to the color of the rug under an object'),
	TypeDesc('blue'),
    TypeDesc('brown'),
    TypeDesc('gray'),
    TypeDesc('green'),
	TypeDesc('orange'),
    TypeDesc('pink'),
	TypeDesc('purple'),
    TypeDesc('red'),
    TypeDesc('tan'),
	TypeDesc('white'),
	TypeDesc('yellow'),
    section_separator_typedesc('Furniture'),
    TypeDesc('bed', can_directly_refer=True),
    TypeDesc('blinds', 'The blinds on the windows'),
    TypeDesc('desk', can_directly_refer=True),
    TypeDesc('desktop'),
    TypeDesc('main_light_switch', 'The main light switch on the wall', can_directly_refer=True),
    TypeDesc('side_table', 'The side table/nightstand next to the bed', can_directly_refer=True),
    TypeDesc('shelf_desk', 'The shelves under the desk'),
    section_separator_typedesc('Large moveable/interactable objects'),
    TypeDesc('book'),
    TypeDesc('chair'),
    TypeDesc('laptop'),
    TypeDesc('pillow'),
	TypeDesc('teddy_bear'),
    section_separator_typedesc('Orientations'),
    TypeDesc('diagonal'),
    TypeDesc('sideways'),
    TypeDesc('upright'),
    TypeDesc('upside_down'),
    section_separator_typedesc('Ramps'),
    TypeDesc('ramp', 'Parent type of all ramp types:'),
    TypeDesc('curved_wooden_ramp'),
    TypeDesc('triangular_ramp'),
	TypeDesc('triangular_ramp_green'),
    TypeDesc('triangular_ramp_tan'),
    section_separator_typedesc('Receptacles'),
    TypeDesc('doggie_bed'),
	TypeDesc('hexagonal_bin'),
    TypeDesc('drawer', 'Either drawer in the side table'),
    TypeDesc('bottom_drawer', 'The bottom of the two drawers in the nightstand near the bed.', can_directly_refer=True),
    TypeDesc('top_drawer', 'The top of the two drawers in the nightstand near the bed.', can_directly_refer=True),
    section_separator_typedesc('Room features'),
    TypeDesc('door', 'The door out of the room', can_directly_refer=True),
    TypeDesc('floor', can_directly_refer=True),
    TypeDesc('mirror', can_directly_refer=True),
    TypeDesc('poster', can_directly_refer=True),
    TypeDesc('room_center', can_directly_refer=True),
    TypeDesc('rug', can_directly_refer=True),
    TypeDesc('shelf'),
    TypeDesc('bottom_shelf', can_directly_refer=True),
    TypeDesc('top_shelf', can_directly_refer=True),
    TypeDesc('sliding_door', 'The sliding doors on the south wall (big windows)'),
    TypeDesc('east_sliding_door', 'The eastern of the two sliding doors (the one closer to the desk)', can_directly_refer=True),
    TypeDesc('west_sliding_door', 'The western of the two sliding doors (the one closer to the bed)', can_directly_refer=True),
    TypeDesc('wall', 'Any of the walls in the room'),
    TypeDesc('east_wall', 'The wall behind the desk', can_directly_refer=True),
    TypeDesc('north_wall', 'The wall with the door to the room', can_directly_refer=True),
    TypeDesc('south_wall', 'The wall with the sliding doors', can_directly_refer=True),
    TypeDesc('west_wall', 'The wall the bed is aligned to', can_directly_refer=True),
    section_separator_typedesc('Small objects'),
	TypeDesc('alarm_clock'),
	TypeDesc('cellphone'),
	TypeDesc('cd'),
	TypeDesc('credit_card'),
	TypeDesc('key_chain'),
	TypeDesc('lamp'),
	TypeDesc('mug'),
	TypeDesc('pen'),
    TypeDesc('pencil'),
	TypeDesc('watch'),
    section_separator_typedesc('Sides'),
    TypeDesc('back'),
    TypeDesc('front'),
    TypeDesc('left'),
    TypeDesc('right'),
)


def extract_predicate_function_args(ast: tatsu.ast.AST) -> typing.List[str]:
    inner = ast
    if 'pred' in ast:
        inner = ast.pred

    elif 'func' in ast:
        inner = ast.func

    if inner is None:
        raise ValueError(f'Could not find predicate or function in {ast}')

    args = []
    arg_index = 1
    arg_key = f'arg_{arg_index}'
    while arg_key in inner and inner[arg_key] is not None:
        term = inner[arg_key].term
        if isinstance(term, tatsu.ast.AST):
            term = term.terminal
        args.append(str(term))
        arg_index += 1
        arg_key = f'arg_{arg_index}'

    return args


def extract_predicate_function_name(ast: tatsu.ast.AST, remove_digits: bool = True):
    if 'pred' in ast:
        rule = ast.pred.parseinfo.rule  # type: ignore
        name = rule.replace('predicate_', '')

    elif 'func' in ast:
        rule = ast.func.parseinfo.rule  # type: ignore
        name = rule.replace('function_', '')

    else:
        rule = ast.parseinfo.rule  # type: ignore
        name = rule.replace('predicate_', '').replace('function_', '')

    if remove_digits:
        if name[-1].isdigit():
            name = name[:-2]

    return name


def extract_n_args(ast: tatsu.ast.AST):
    return len(extract_predicate_function_args(ast))


def wrapped_extract_n_args(ast: tatsu.ast.AST, key: typing.Optional[str] = None):
    return ('n_args', len(extract_predicate_function_args(ast)))


PREDICATE_RULES = {
    PREDICATE: (extract_predicate_function_name, (wrapped_extract_n_args, ))
}

FUNCTION_RULES = {
    FUNCTION: (extract_predicate_function_name, (wrapped_extract_n_args, ))
}


def extract_single_variable_type(ast):
    if 'var_type' in ast and 'type' in ast.var_type and isinstance(ast.var_type.type, tatsu.ast.AST):
        return ast.var_type.type.terminal

    return None


def extract_either_variable_types(ast):
    if 'type_names' in ast:
        if isinstance(ast.type_names, str):
            return (ast.type_names, )

        if isinstance(ast.type_names, tatsu.ast.AST):
            return (ast.type_names.terminal,)

        return tuple(t.terminal for t in ast.type_names)

    return None


def extract_pref_name_and_types(ast):
    if 'object_types' in ast:
        if ast.object_types is None:
            return None

        if isinstance(ast.object_types, tatsu.ast.AST):
            return ast.object_types.type_name.terminal
        else:
            return [t.type_name.terminal for t in ast.object_types if 'type_name' in t]

    return None


def extract_types_from_predicates_and_functions(ast):
    terms = extract_predicate_function_args(ast)
    return [term for term in terms if not term.startswith('?')]


def extract_co_ocurring_types(ast, key):
    if 'type_names' in ast:
        if not isinstance(ast.type_names, (str, tatsu.ast.AST)):
            types = list(t.terminal for t in ast.type_names)
            if key in types:
                types.remove(key)

            return ('co_ocurring_types', types)

    return (None, None)


TYPE_RULES = {
    'variable_type_def': (extract_single_variable_type, None),
    'color_variable_type_def': (extract_single_variable_type, None),
    'orientation_variable_type_def': (extract_single_variable_type, None),
    'side_variable_type_def': (extract_single_variable_type, None),
    'either_types': (extract_either_variable_types, None),
    'either_color_types': (extract_either_variable_types, None),
    'either_orientation_types': (extract_either_variable_types, None),
    'either_side_types': (extract_either_variable_types, None),
    'pref_name_and_types': (extract_pref_name_and_types, None),
    'predicate': (extract_types_from_predicates_and_functions, None),
    'function': (extract_types_from_predicates_and_functions, None),
}

SCORING_CONSIDER_USED_RULES = (
    'scoring_maximize', 'scoring_minimize',
    'preference-eval', # 'count_maximal_nonoverlapping',
    'count_once_per_external_objects', 'scoring_neg_expr', 'pref_object_type',
    'pref_name_and_types', 'count', 'count_once_per_objects',
    'count_once',
)


def main(args):
    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    parser = DSLToLatexParser(args.template_file, args.output_file, args.new_data_start)

    pref_consider_described_rules = list(chain.from_iterable([block_rules for block_description, block_rules in SHARED_BLOCKS.values()])) + ['super_predicate', 'super_predicate_and', 'super_predicate_or', 'super_predicate_not', 'super_predicate_forall', 'super_predicate_exists']

    setup_translator = SectionTranslator(SETUP_SECTION_KEY, SETUP_BLOCKS, (SHARED_BLOCKS[FUNCTION_COMPARISON], SHARED_BLOCKS[VARIABLE_LIST], SHARED_BLOCKS[PREDICATE]),
        consider_used_rules=['setup_not', 'setup_statement', 'setup_or'])
    pref_translator = SectionTranslator(PREFERENCES_SECTION_KEY, PREFERENCES_BLOCKS, section_name='Gameplay Preferences',
        consider_described_rules=pref_consider_described_rules)  # additional_blocks=(SHARED_BLOCKS[FUNCTION_COMPARISON], SHARED_BLOCKS[VARIABLE_LIST], SHARED_BLOCKS[PREDICATE])
    terminal_translator = SectionTranslator(TERMINAL_SECTION_KEY, TERMINAL_BLOCKS, None, section_name='Terminal Conditions', consider_used_rules=['terminal_not'])
    scoring_translator = SectionTranslator(SCORING_SECTION_KEY, SCORING_BLOCKS, None, consider_used_rules=SCORING_CONSIDER_USED_RULES)

    non_grammar_section = FakeTranslator(r"\section{Non-Grammar Definitions}")
    predicate_translator = RuleTypeTranslator(PREDICATES_SECTION_KEY, PREDICATE_RULES, predicate_data_to_lines, descriptions=PREDICATE_DESCRIPTIONS)
    function_translator = RuleTypeTranslator(FUNCTIONS_SECTION_KEY, FUNCTION_RULES, predicate_data_to_lines, descriptions=FUNCTION_DESCRIPTIONS)
    type_translator = RuleTypeTranslator(TYPES_SECTION_KEY, TYPE_RULES, type_data_to_lines, descriptions=TYPE_DESCRIPTIONS)

    parser.register_processor(setup_translator)
    parser.register_processor(pref_translator)
    parser.register_processor(terminal_translator)
    parser.register_processor(scoring_translator)

    parser.register_processor(non_grammar_section)
    parser.register_processor(predicate_translator)
    parser.register_processor(function_translator)
    parser.register_processor(type_translator)

    for test_file in args.test_files:
        for ast in cached_load_and_parse_games_from_file(test_file, grammar_parser, False):
            parser(ast)

    parser.process()
    parser.output()

    if args.compile_pdf:
        out_dir, original_out_file = os.path.split(args.output_file)
        out_dir = os.path.join(out_dir, 'out')
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, original_out_file)
        shutil.copy(args.output_file, out_file)

        os.system(f'cd {out_dir} && pdflatex -shell-escape {original_out_file}')
        pdf_out_path = os.path.splitext(out_file)[0] + '.pdf'
        pdf_final_path = args.output_file.replace('.tex', '.pdf')
        shutil.copy(pdf_out_path, pdf_final_path)


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)

    main(args)
