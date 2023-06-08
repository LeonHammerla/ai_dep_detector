import math
import os
import sys
from collections import Counter
from typing import Optional, Tuple, List, Dict, Union, Any, Callable
import torch
from treelib import Tree, Node
from operator import indexOf
import spacy
from spacy.tokens import Doc
from spacy.language import Language
from nltk import word_tokenize
import numpy as np
import sklearn

@Language.component("special_split")
def special_split(doc):
    # Note this code is not robust with handling indices
    for i, tok in enumerate(doc):
        if i == 0:
            tok.is_sent_start = True
        else:
            tok.is_sent_start = False
    return doc


class DependencyFeatureExtractor:
    def __init__(self, lang: str = "en", funct_analysis: bool = False):
        self.lang = lang
        self.lang_dict = {"en": "en_core_web_sm"}
        self.model = spacy.load(self.lang_dict[self.lang])
        self.model.add_pipe("special_split", before="parser")
        self.dep_labels = self.model.get_pipe("parser").labels
        self.softmax = torch.nn.Softmax(dim=0)
        self.softmax_batch = torch.nn.Softmax(dim=1)
        self.funct_analysis = funct_analysis

    @staticmethod
    def n_verbs_in_sentence(pos: List[str]) -> int:
        return pos.count("VERB")

    @staticmethod
    def n_token_in_sentence(token: List[str]) -> int:
        """
        Calculates how many tokens are in a sentence.
        :param tokens:
        :return:
        """
        return len(token)

    @staticmethod
    def mdd_of_sent(doc: Doc) -> float:
        """
        For a Pair of dependent and governor the distance of their position
        in a sentence is calculated.
        The mdd ist the mean for all those pairs in a sentence.

        Definition:
        MDD(sent) = 1 / (n - 1) * sum(DD:0...DD:n-1)

        Here n is the number of words in the sentence and DDi is the dependency distance of the
        i-th syntactic link of the sentence. Usually in a sentence there is one word (the root verb)
        without a head, whose DD is defined as zero.
        :param dependencies:
        :param tokens:
        :return:
        """
        position_sum = 0
        length = 0
        for tok in doc:
            position_sum += abs(tok.i - tok.head.i)
            length += 1

        # print(position_sum)
        return position_sum / (length - 1) if (length - 1) > 0 else 0.0

    @staticmethod
    def calc_max_dep_depth_of_sentence_and_depthmap(dep_tree: Tree) -> Tuple[
        int, Dict[int, int]]:
        """
        Function calculates max depth of dependency tree of a sentence and returnes a dict with a count of every d
        :param cas:
        :param dependencies:
        :return:
        """

        depth_list = []
        for node in dep_tree.all_nodes():
            depth = dep_tree.depth(node)
            depth_list.append(depth)
        return max(depth_list), dict(Counter(depth_list))

    @staticmethod
    def node_path_dist(node1: Node,
                       node2: Node,
                       tree: Tree) -> Union[int, float]:
        """
        Function calculates the geodesic distance for two nodes in a
        given dependency tree. If nodes are not connected in the tree, distance
        becomes +inf.
        :param node1:
        :param node2:
        :param tree:
        :return:
        """
        name1 = node1.identifier
        name2 = node2.identifier
        if tree.is_ancestor(name1, name2):
            return tree.level(name2) - tree.level(name1)
        elif tree.is_ancestor(name2, name1):
            # return tree.level(name1) - tree.level(name2)
            return math.inf
        else:
            return math.inf

    @staticmethod
    def make_dist_dict(_Vs: List[Node],
                       dep_tree: Tree) -> Dict[Node, Dict[Union[int, float], List[Node]]]:
        """
        Function for creating a dict containing every path distances for all vertices
        to all other vertices.
        Dict keys are Nodes, and values are dicts. For these dicts keys are int or float and value
        list of nodes.
        :param _Vs:
        :param dep_tree:
        :return:
        """
        dist_dict = {}
        for i in _Vs:
            a = []
            for j in _Vs:
                a.append((DependencyFeatureExtractor.node_path_dist(i, j, dep_tree), j))
            b = dict()
            for j in a:
                if j[0] in b:
                    b[j[0]].append(j[-1])
                else:
                    b[j[0]] = [j[-1]]
            dist_dict[i] = b
        return dist_dict

    @staticmethod
    def out_degree(node: Node, dist_dict: Dict[Node, Dict[Union[int, float], List[Node]]]) -> int:
        """
        Function returns the out-degree of a given node in a dependency-tree.
        :param node:
        :param dist_dict:
        :return:
        """
        try:
            return len(dist_dict[node][1])
        except:
            return 0

    @staticmethod
    def get_absolute_prestige(dist_dict: Dict[Node, Dict[Union[int, float], List[Node]]]):
        """
        Function for calculating the absolute prestige:
        https://www.researchgate.net/publication/200110853_Structural_analysis_of_hypertexts_Identifying_hierarchies_and_useful_
        Prestige for node is status - contrastatus. Absolute Prestige is sum of abs(node-prestige).
        :param dist_dict:
        :return:
        """
        node_prestiges = []
        for node in dist_dict:
            status = 0
            for dist in dist_dict[node]:
                if 0 < dist < math.inf:
                    status += (dist * len(dist_dict[node][dist]))
            contra_status = 0
            for node2 in dist_dict:
                for dist2 in dist_dict[node2]:
                    if node in dist_dict[node2][dist2]:
                        if 0 < dist2 < math.inf:
                            contra_status += dist2
                        break
            prestige_of_node = status - contra_status
            node_prestiges.append(abs(prestige_of_node))
        absolute_prestige = sum(node_prestiges)
        return absolute_prestige

    @staticmethod
    def make_ssl(dist_dict: Dict[Node, Dict[Union[int, float], List[Node]]],
                 root_node: Node,
                 _L: List[Node]) -> Dict[Union[int, float], List[Node]]:
        """
        Function for getting the set of subset of leaves ordered by their distance to
        the root node of the tree.
        :param dist_dict:
        :param root_node:
        :param _L:
        :return:
        """
        ssl = dict()
        for dist in dist_dict[root_node]:
            leaves = []
            for leave in _L:
                if leave in dist_dict[root_node][dist]:
                    leaves.append(leave)
            if leaves:
                ssl[dist] = leaves
        return ssl

    @staticmethod
    def get_lde(ssl: Dict[Union[int, float], List[Node]],
                _L: List[Node]) -> float:
        """
        Function for calculating LDE (Leaf Depth/Distance Entropy)
        :param ssl:
        :param _L:
        :return:
        """
        total_number_of_subsets = len(ssl.keys())
        if total_number_of_subsets > 1:
            lde_sum = 0
            for dist in ssl:
                pi = len(ssl[dist]) / len(_L) if len(_L) > 0 else 0.0
                lde_sum += (pi * math.log2(pi))
            lde_sum *= -1
            lde = lde_sum / math.log2(total_number_of_subsets) if math.log2(total_number_of_subsets) > 0 else 0.0
            return lde
        else:
            return 0.0

    @staticmethod
    def make_sse(tokens: List[str],
                 arcs: Dict[str, list],
                 _js: Callable[[Node], int],
                 dep_tree: Tree) -> Dict[int, List[Tuple[int, int]]]:
        """
        Function constructs the set of subsets with one subset containing
        all dependency arcs between two dependencys with the same distance (position-wise).
        :param tokens:
        :param arcs:
        :param _js:
        :param dep_tree:
        :return:
        """
        sse = dict()
        for i in range(-len(tokens), len(tokens) + 1):
            edges = []
            for edge in arcs["edges"]:
                if _js(dep_tree.get_node(edge[0])) - _js(dep_tree.get_node(edge[1])) == i:
                    edges.append(edge)
            if edges:
                sse[i] = edges
        return sse

    @staticmethod
    def get_dde(sse: Dict[int, List[Tuple[int, int]]],
                arcs: Dict[str, list]) -> float:
        """
        Function for calculating DDE (dependency distance entropy).
        :param sse:
        :param arcs:
        :return:
        """
        total_number_of_subsets = len(sse.keys())
        total_number_of_arcs = len(arcs["edges"])
        if total_number_of_subsets > 1:
            dde_sum = 0
            for dist in sse:
                pi = len(sse[dist]) / total_number_of_arcs if total_number_of_arcs > 0 else 0.0
                dde_sum += (pi * math.log2(pi))
            dde_sum *= -1
            dde = dde_sum / math.log2(total_number_of_subsets) if math.log2(total_number_of_subsets) > 0 else 0.0
            return dde
        else:
            return 0.0

    @staticmethod
    def get_highest_level_predecessor(node1: Node, node2: Node, dep_tree: Tree) -> Node:
        """
        Function for finding highest level predecessor for two
        given nodes in their tree.
        :param node1:
        :param node2:
        :param dep_tree:
        :return:
        """
        tree_id = dep_tree.identifier
        predecessors_node1 = []
        if node1.is_root(tree_id) or node2.is_root(tree_id):
            return dep_tree.get_node(dep_tree.root)
        else:
            while True:
                predecessor = node1.predecessor(tree_id)
                if predecessor is None:
                    break
                else:
                    predecessors_node1.append(predecessor)
                    node1 = dep_tree.get_node(predecessor)

            predecessors_both = []
            while True:
                predecessor = node2.predecessor(tree_id)
                if predecessor is None:
                    break
                else:
                    if predecessor in predecessors_node1:
                        predecessors_both.append(predecessor)
                    node2 = dep_tree.get_node(predecessor)

            final_predecessor = ("", -1)
            for predecessor in predecessors_both:
                depth = dep_tree.depth(predecessor)
                if depth > final_predecessor[-1]:
                    final_predecessor = (predecessor, depth)

            return dep_tree.get_node(final_predecessor[0])

    @staticmethod
    def get_all_leave_pairs(_L: List[Node]) -> List[Tuple[Node, Node]]:
        """
        This Function returns every possible leave pair.
        :param _L:
        :return:
        """
        leave_pairs = [(a, b) for idx, a in enumerate(_L) for b in _L[idx + 1:]]
        return leave_pairs

    @staticmethod
    def get_tci_and_l(_L: List[Node],
                      all_leave_pairs: List[Tuple[Node, Node]],
                      dist_dict: Dict[Node, Dict[Union[int, float], List[Node]]],
                      dep_tree: Tree) -> Tuple[int, int]:
        """
        Function for getting the Total Cophenetic Index for calculating later the
        imbalance of tree.
        :param _L:
        :param all_leave_pairs:
        :param dist_dict:
        :param dep_tree:
        :return:
        """
        # ==== Calculation of tci (Total Cophenetic Index) ====
        # --> Calculating k:
        k = 0
        for node in dist_dict:
            if DependencyFeatureExtractor.out_degree(node, dist_dict) == 1:
                k += 1
        # --> Calculating expected_number_of_leaves_in_caterpillar_graph:
        expected_number_of_leaves_in_caterpillar_graph = len(_L) + k
        # --> Calculation of TCI:
        tci_sum = 0
        if expected_number_of_leaves_in_caterpillar_graph > 3:
            for pair in all_leave_pairs:
                tci_sum += dep_tree.depth(
                    DependencyFeatureExtractor.get_highest_level_predecessor(pair[0], pair[-1], dep_tree))
            return tci_sum, expected_number_of_leaves_in_caterpillar_graph
        else:
            return 0, expected_number_of_leaves_in_caterpillar_graph

    @staticmethod
    def get_imbalance(tci: int, l: int) -> float:
        """
        Function for calculating imbalance index of tree.
        :param tci:
        :param l:
        :return:
        """
        if l > 3:
            return tci / math.comb(l, 3)
        else:
            return 0.0

    @staticmethod
    def get_hirsch_index(dist_dict: Dict[Node, Dict[Union[int, float], List[Node]]],
                         dep_tree: Tree,
                         root_node: Node) -> int:
        """
        Function for calculating the hirsch index.
        :param dist_dict:
        :param dep_tree:
        :param root_node:
        :return:
        """
        dist_to_width_from_root = dict()
        for dist in dist_dict[root_node]:
            dist_to_width_from_root[dist] = len(dist_dict[root_node][dist])

        possible_indices = [0]
        for i in range(1, dep_tree.depth() + 1):
            is_true = True
            for j in range(1, i + 1):
                if dist_to_width_from_root[j] - 1 >= j:
                    pass
                else:
                    is_true = False
                    break
            if is_true:
                possible_indices.append(i)

        return max(possible_indices)

    @staticmethod
    def get_ratio_hirsch_index(hirsch_index: int,
                               root_node: Node,
                               dist_dict: Dict[Node, Dict[Union[int, float], List[Node]]],
                               _Vs: List[Node]) -> float:
        """
        Function for calculating ratio of vertices contributing to h-index.
        :param hirsch_index:
        :param dep_tree:
        :param root_node:
        :param dist_dict:
        :param _Vs:
        :return:
        """
        node_sum = 1  # for root
        for dist in range(1, hirsch_index + 1):
            node_sum += len(dist_dict[root_node][dist])

        return node_sum / len(_Vs) if len(_Vs) > 0 else 0.0

    @staticmethod
    def complexity_analysis(dist_dict: Dict[Node, Dict[Union[int, float], List[Node]]],
                            root_node: Node) -> Tuple[float, int, int]:
        """
        Function returns different measurements regarding the complexity:
        (r=root, A=edges, V=Vertices, w=elem of V)
        - complexity ratio: |{w | (r, w) elem of A}| / |V|
        - absolute complexity: |{w | (r, w) elem of A}|
        - order of tree:
        :param dist_dict:
        :param dep_tree:
        :return:
        """
        # ==== Calculating different measurements for complexity ====
        root_degree = DependencyFeatureExtractor.out_degree(root_node, dist_dict)
        # --> complexity ratio:
        complexity_ratio = root_degree / len(list(dist_dict.keys())) if len(list(dist_dict.keys())) > 0 else 0.0
        # --> absolute complexity:
        absolute_complexity = root_degree
        # --> order of tree:
        order_T = len(list(dist_dict.keys()))

        return complexity_ratio, absolute_complexity, order_T

    @staticmethod
    def dependency_analysis(dep_tree: Tree,
                            _Vs: List[Node],
                            root_node: Node,
                            dist_dict: Dict[Node, Dict[Union[int, float], List[Node]]]) -> Tuple[float, float]:
        """
        Function for dependency analysis.
        - dependency index
        - stratum of tree
        :param dep_tree:
        :param _Vs:
        :param root_node:
        :param dist_dict:
        :return:
        """
        # ==== Calculating different measurements related to dependency ====
        order_T = len(_Vs)

        # --> dependency index (altmann index):
        dependency_index_sum = 0
        for i in range(1, dep_tree.depth() + 1):
            try:
                dependency_index_sum += (i * len(dist_dict[root_node][i - 1]))
            except:
                pass
        dependency_index = (2 * dependency_index_sum) / (order_T * (order_T + 1))
        # --> stratum of tree (absolute-prestige/LAP):
        lap = (order_T ** 3) / 4 if order_T % 2 == 0 else (order_T ** 3 - order_T) / 4
        try:
            stratum_of_tree = DependencyFeatureExtractor.get_absolute_prestige(dist_dict) / lap
        except:
            stratum_of_tree = 0

        return dependency_index, stratum_of_tree

    @staticmethod
    def depth_analysis(dep_tree: Tree,
                       _Vs: List[Node],
                       root_node: Node,
                       dist_dict: Dict[Node, Dict[Union[int, float], List[Node]]],
                       _L: List[Node]) -> Tuple[int, float, float, float]:
        """
        Function for calculating different depth related measurements:
        - depth of tree
        - ratio of vertices on longest path starting from root
        - leaf distance entropy
        - ratio of leaves at distance one to root
        :param dep_tree:
        :param _Vs:
        :param root_node:
        :param dist_dict:
        :param _L:
        :return:
        """
        # ==== Depth related measurements of tree ====
        order_T = len(_Vs)

        # --> depth of tree:
        depth_T = dep_tree.depth()
        # --> ratio of vertices on longest path starting from root:
        _T_T = depth_T / order_T if order_T > 0 else 0.0
        # --> leaf distance entropy (ssl = set of subset of leaves):
        ssl = DependencyFeatureExtractor.make_ssl(dist_dict, root_node, _L)
        lde = DependencyFeatureExtractor.get_lde(ssl, _L)
        # --> ratio of leaves at distance one to root:
        try:
            _L1 = len(ssl[1]) / max(1, order_T)
        except:
            _L1 = 0.0

        return depth_T, _T_T, lde, _L1

    @staticmethod
    def distance_analysis(dep_tree: Tree,
                          _Vs: List[Node],
                          doc: Doc,
                          tokens: List[str],
                          _js: Callable[[Node], int],
                          arcs: Dict[str, list]) -> Tuple[float, float, float, float]:
        """
        Function for calculating different dependency distance related measurements:
        - mean dependency distance
        - dependency distance entropy (sse = set of subsets entropy)
        - ratio of arcs between adjacent tokens
        - ratio of arcs manifesting distances occurring once
        :param dep_tree:
        :param _Vs:
        :param tokens:
        :param dependencies:
        :param _js:
        :param arcs:
        :return:
        """
        # ==== Calculate some distance related measurements ====
        # --> mean dependency distance:
        mdd = DependencyFeatureExtractor.mdd_of_sent(doc)
        # --> dependency distance entropy (sse = set of subsets entropy):
        sse = DependencyFeatureExtractor.make_sse(tokens, arcs, _js, dep_tree)
        dde = DependencyFeatureExtractor.get_dde(sse, arcs)
        # --> ratio of arcs between adjacent tokens:
        try:
            _D1 = len(sse[1]) / max(1, len(arcs["edges"]))
        except:
            _D1 = 0.0
        # --> ratio of arcs manifesting distances occurring once:
        arcs_manifesting_distances_occurring_once = 0
        for dist in sse:
            if len(sse[dist]) == 1:
                arcs_manifesting_distances_occurring_once += 1
        _D_sets_1 = arcs_manifesting_distances_occurring_once / max(1, len(
            arcs["edges"])) if arcs_manifesting_distances_occurring_once > 0 else 0.0

        return mdd, dde, _D1, _D_sets_1

    @staticmethod
    def imbalance_analysis(_L: List[Node],
                           dist_dict: Dict[Node, Dict[Union[int, float], List[Node]]],
                           dep_tree: Tree) -> float:
        """
        Function for gettling imbalance related measurements:
        - imbalance index
        :param _L:
        :param dist_dict:
        :param dep_tree:
        :return:
        """
        # ==== Different Imbalance related Measurements ====
        # --> Total Cophenetic Index:
        leave_pairs = DependencyFeatureExtractor.get_all_leave_pairs(_L)
        tci, l = DependencyFeatureExtractor.get_tci_and_l(_L, leave_pairs, dist_dict, dep_tree)
        # --> imbalance Index of tree:
        imbalance_index = DependencyFeatureExtractor.get_imbalance(tci, l)

        return imbalance_index

    @staticmethod
    def length_analysis(_L: List[Node],
                        _Vs: List[Node]) -> Tuple[float, int]:
        """
        Function returns two leaf based measurements:
        - ratio of leaves
        - number of leaves
        :param _L:
        :param _Vs:
        :return:
        """
        # ==== Length related measurements ====
        # --> number of leaves:
        number_of_leaves = len(_L)
        # --> ratio of leaves:
        ratio_of_leaves = number_of_leaves / len(_Vs) if len(_Vs) > 0 else 0.0

        return ratio_of_leaves, number_of_leaves

    @staticmethod
    def functional_analysis(dep_labels: Tuple[str], arcs: Dict[str, list]) -> Tuple[List[float], List[float]]:
        """
        Functional analysis.
        - ratio of arcs of type X
        - number of arcs of type X

        MATE-Parser Labels used:
        CJ - (conjunct),
        CP - (complementizer),
        DA - (dative),
        HD - (head),
        MO - (modifier),
        NK - (negation),
        OA - (accusative object),
        OA2- (second accusative object),
        OC - (clausal object),
        PD - (predicate),
        RC - (relative clause),
        SB - (subject).
        :param arcs:
        :return:
        """
        # dep_labels = ["CJ", "CP", "DA", "HD", "MO", "NK", "OA", "OA2", "OC", "PD", "RC", "SB"]
        # universal_dep_labels = ["CONJ", "CCOMP", "", "--", "", "NEG", ]
        # ==== Some measurements for a functional dependency analysis ====
        # --> ratio of arcs of type X and number of arcs of type X:
        total_number_of_arcs = max(1, len(arcs["edges"]))
        scores_ratio = []
        scores_total_number = []
        for dep_label in dep_labels:
            label_count = 0
            for label in arcs["labels"]:
                if label == dep_label:
                    label_count += 1
            scores_ratio.append(label_count / total_number_of_arcs if total_number_of_arcs > 0 else 0.0)
            scores_total_number.append(label_count)
        return scores_ratio, scores_total_number

    @staticmethod
    def width_analysis(dist_dict: Dict[Node, Dict[Union[int, float], List[Node]]],
                       root_node: Node,
                       dep_tree: Tree,
                       _Vs: List[Node]):
        """
        Function for the width analysis. Containing the following measurements:
        - width of tree
        - the lowest level of maximum width
        - ratio of vertices belonging to the latter level
        - Hirsch index by level (h-index)
        - ratio of vertices contributing to h-index
        - relative h-index
        :param dist_dict:
        :param root_node:
        :param dep_tree:
        :param _Vs:
        :return:
        """
        # ==== Analysis of the width structure of the tree ====
        width_max = [0, 0]
        for dist in dist_dict[root_node]:
            if len(dist_dict[root_node][dist]) == width_max[0]:
                width_max[1] = min(width_max[1], dist)
            elif len(dist_dict[root_node][dist]) > width_max[0]:
                width_max[0] = len(dist_dict[root_node][dist])
                width_max[1] = dist
            else:
                pass
        # --> width of tree:
        width = width_max[0]
        # --> lowest level of maximum width:
        lowest_level_of_max_width = width_max[1]
        # --> ratio of vertices belonging to the latter level:
        ratio_of_v_at_level = len(dist_dict[root_node][lowest_level_of_max_width]) / len(_Vs) if len(_Vs) > 0 else 0.0
        # --> Hirsch index by level:
        hirsch_index = DependencyFeatureExtractor.get_hirsch_index(dist_dict, dep_tree, root_node)
        # --> ratio of vertices contributing to h-index:
        ratio_of_hirsch_index = DependencyFeatureExtractor.get_ratio_hirsch_index(hirsch_index, root_node, dist_dict,
                                                                                  _Vs)
        # --> relative h-index
        relative_h_index = hirsch_index / dep_tree.depth() if dep_tree.depth() > 0 else 0.0

        return width, lowest_level_of_max_width, ratio_of_v_at_level, hirsch_index, ratio_of_hirsch_index, relative_h_index

    @staticmethod
    def create_dependency_tree(doc: Doc) -> Tuple[Tree,
                                                  Dict[str, Union[Union[Callable[[Node], int], Dict[
                                                      str, List[Any]], Callable[
                                                                            [Tuple[int, int]], str]], Any]],
                                                  Dict[str, Union[Union[Dict[Node, Dict[
                                                      Union[int, float], List[Node]]], Callable[
                                                                            [Node, Dict[Node, Dict[
                                                                                Union[int, float], List[
                                                                                    Node]]]], int], Callable[
                                                                            [Node, Node, Tree], Union[
                                                                                int, float]]], Any]]]:
        """
        Function creates dependency tree for given Sentence.
        :param tokens:
        :param dependencies:
        :return:
        """

        # ==== Creating Dependency-Tree ====
        arcs = {"edges": [], "labels": []}
        dep_tree = Tree()
        # --> Collecting Edges and their labels
        for tok in doc:
            label = tok.dep_
            edge = (tok.head.i, tok.i)
            arcs["edges"].append(edge)
            arcs["labels"].append(label)
        # --> Finding edge and removing it, because it should not be present in tree
        for i in range(0, len(arcs["labels"])):
            if (arcs["labels"][i] == "--") or (arcs["labels"][i].lower() == "root"):
                root_index = arcs["edges"][i][-1]
                del arcs["edges"][i]
                del arcs["labels"][i]
                break
        # --> Adding root to tree-object
        dep_tree.create_node(doc[root_index].text, root_index, data=doc[root_index])
        # --> adding all remaining nodes to tree with while-loop
        cur_idx = [root_index]
        while True:
            cur_childs = []
            idx = cur_idx[0]
            for i in range(0, len(arcs["edges"])):
                if idx == arcs["edges"][i][0]:
                    cur_childs.append(arcs["edges"][i])
            if not cur_childs:
                cur_idx = cur_idx[1:]
                if not cur_idx:
                    break
                else:
                    pass
            else:
                for j in cur_childs:
                    cur_idx.append(j[-1])
                    dep_tree.create_node(doc[j[-1]].text, j[-1], parent=j[0], data=doc[j[-1]])
                cur_idx = cur_idx[1:]
        # dep_tree.show()

        # ==== Tree consists of Tuple: T(S) = (Vs, As, ls, js, rs) ====
        # --> Vs : Vertices
        _Vs = dep_tree.all_nodes()
        # --> rs : root of dependency-tree
        _rs = dep_tree.get_node(dep_tree.root)
        # --> As : Edges of dependency-tree
        _As = arcs
        # --> js : projection function for pos of token in sentence is: js(wi) = i
        _js: Callable[[Node], int] = lambda x: x.identifier
        # --> ls : arc labeling function for as element of As: ls(as) = arc(as)
        _ls: Callable[[Tuple[int, int]], str] = lambda x: _As["labels"][indexOf(_As["edges"], x)]

        # ==== complete tree-components: ====
        dep_tree_approx = {"Vs": _Vs, "As": _As, "js": _js, "ls": _ls, "rs": _rs}

        # ==== Additional tree-components ====
        # --> Leaves
        _L = dep_tree.leaves()
        # --> geodesic distance for two nodes in dep-tree
        _d = DependencyFeatureExtractor.node_path_dist
        # --> sets with all vertices and their distance
        _dist_dict = DependencyFeatureExtractor.make_dist_dict(_Vs, dep_tree)
        # --> function to determine out-degree of a vertex
        _degree = DependencyFeatureExtractor.out_degree

        # ==== complete additional tree-components ====
        dep_tree_approx_add = {"L": _L, "d": _d, "dist_dict": _dist_dict, "degree": _degree}

        return dep_tree, dep_tree_approx, dep_tree_approx_add

    def combine_all_measurements(self, sentence: str) -> list:
        """
        Function collects all available measurements for one sentence.
        :param cas:
        :param sentence:
        :return:
        """

        doc = self.model(sentence)

        ltoken = []
        ldependencies = []
        lpos = []
        ltag = []
        for token in doc:
            ltoken.append(token.text)
            ldependencies.append(token.dep_)
            lpos.append(token.pos_)
            ltag.append(token.tag_)

        # ==== Making Dependency Tree with all desired features ====
        # --> components --> {"Vs": _Vs, "As": _As, "js": _js, "ls": _ls, "rs": _rs}
        # --> additional_components --> {"L": _L, "d": _d, "dist_dict": _dist_dict, "degree": _degree}
        dep_tree, components, additional_components = self.create_dependency_tree(doc)

        # ==== Getting all Measurements ====
        all_measurements = []
        # --> n_verbs 0)
        n_verbs = DependencyFeatureExtractor.n_verbs_in_sentence(lpos)
        all_measurements.append(n_verbs)
        # ==== Complexity Analysis ====
        complexity_results = DependencyFeatureExtractor.complexity_analysis(
            dist_dict=additional_components["dist_dict"],
            root_node=components["rs"])
        # --> Complexity ratio 1):
        all_measurements.append(complexity_results[0])
        # --> absolute Complexity 2):
        all_measurements.append(complexity_results[1])
        # --> order of tree 3):
        all_measurements.append(complexity_results[2])
        # ==== Dependency Analysis ====
        dependency_results = DependencyFeatureExtractor.dependency_analysis(dep_tree=dep_tree,
                                                                            _Vs=components["Vs"],
                                                                            root_node=components["rs"],
                                                                            dist_dict=additional_components[
                                                                                "dist_dict"])
        # --> altmann index (dependency index) 4):
        all_measurements.append(dependency_results[0])
        # --> stratum of tree 5):
        all_measurements.append(dependency_results[1])
        # ==== Depth Analysis ====
        depth_results = DependencyFeatureExtractor.depth_analysis(dep_tree=dep_tree,
                                                                  _Vs=components["Vs"],
                                                                  root_node=components["rs"],
                                                                  dist_dict=additional_components["dist_dict"],
                                                                  _L=additional_components["L"])
        # --> depth of tree 6):
        all_measurements.append(depth_results[0])
        # --> ratio of vertices on longest path starting from root 7):
        all_measurements.append(depth_results[1])
        # --> leaf distance entropy 8):
        all_measurements.append(depth_results[2])
        # --> ratio of leaves at distance one to root 9):
        all_measurements.append(depth_results[3])
        # ==== Distance Analysis ====
        distance_results = DependencyFeatureExtractor.distance_analysis(dep_tree=dep_tree,
                                                                        _Vs=components["Vs"],
                                                                        tokens=ltoken,
                                                                        doc=doc,
                                                                        _js=components["js"],
                                                                        arcs=components["As"])
        # --> mean dependency distance 10)
        all_measurements.append(distance_results[0])
        # --> dependency distance entropy 11):
        all_measurements.append(distance_results[1])
        # --> ratio of arcs between adjacent tokens 12):
        all_measurements.append(distance_results[2])
        # --> ratio of arcs manifesting distances occurring once 13):
        all_measurements.append(distance_results[3])
        # ==== Imbalance Analysis ====
        imbalance_results = DependencyFeatureExtractor.imbalance_analysis(_L=additional_components["L"],
                                                                          dist_dict=additional_components["dist_dict"],
                                                                          dep_tree=dep_tree)
        # --> Imbalance Index 14):
        all_measurements.append(imbalance_results)
        # ==== Length Analysis ====
        length_results = DependencyFeatureExtractor.length_analysis(_L=additional_components["L"],
                                                                    _Vs=components["Vs"])
        # --> ratio of leaves 15):
        all_measurements.append(length_results[0])
        # --> number of leaves 16):
        all_measurements.append(length_results[1])

        if self.funct_analysis:
            # ==== Function Analysis ====
            function_results = DependencyFeatureExtractor.functional_analysis(self.dep_labels, arcs=components["As"])
            # --> ratio of arcs of type X 17):
            for x in function_results[0]:
                all_measurements.append(x)
            # --> number of arcs of type X 18):
            for x in function_results[1]:
                all_measurements.append(x)

        # ==== Width Analysis ====
        width_results = DependencyFeatureExtractor.width_analysis(dist_dict=additional_components["dist_dict"],
                                                                  root_node=components["rs"],
                                                                  dep_tree=dep_tree,
                                                                  _Vs=components["Vs"])
        # --> width of tree 19):
        all_measurements.append(width_results[0])
        # --> the lowest level of maximum width 20):
        all_measurements.append(width_results[1])
        # --> ratio of vertices belonging to the latter level 21):
        all_measurements.append(width_results[2])
        # --> Hirsch index by level (h-index) 22):
        all_measurements.append(width_results[3])
        # --> ratio of vertices contributing to h-index 23):
        all_measurements.append(width_results[4])
        # --> relative h-index 24):
        all_measurements.append(width_results[5])

        all_measurements = [0 if math.isnan(x) else x for x in all_measurements]
        return all_measurements

    @staticmethod
    def normalize_data(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def __call__(self, sent: str, *args, **kwargs):
        # 113 with funct, else 23
        # print(sent)
        return torch.Tensor(self.normalize_data(self.combine_all_measurements(sentence=sent)))



if __name__ == "__main__":

    extractor = DependencyFeatureExtractor()
    print(extractor("We find only very small changes in the gas velocities reflected by either of the line profiles, no significant changes in the flux to continuum ratios, and hence no significant changes in the [Fe II]/[S I] flux ratios."))

