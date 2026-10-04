from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple, Union


TRUE_TAG = "⊤"
FALSE_TAG = "⊥"

VertexId = Hashable
FeatureId = Union[int, str]
LeafLabel = Union[bool, int, str]


def _normalize_leaf_label(label: LeafLabel) -> LeafLabel:
    if label is True or label in (TRUE_TAG, "T", "True", "true"):
        return TRUE_TAG
    if label is False or label in (FALSE_TAG, "F", "False", "false"):
        return FALSE_TAG
    return label


def _is_boolean_leaf_label(label: LeafLabel) -> bool:
    return _normalize_leaf_label(label) in (TRUE_TAG, FALSE_TAG)


@dataclass
class MDDVertex:
    """Vertex of the explicit MDD representation used in the paper."""

    vertex_id: VertexId
    number: Optional[Any] = None

    def is_leaf(self) -> bool:
        return isinstance(self, MDDLeafVertex)


@dataclass
class MDDDecisionVertex(MDDVertex):
    """
    Internal MDD vertex.
    """

    feature_id: FeatureId = 0
    children: Dict[int, VertexId] = field(default_factory=dict)

    def get_feature_id(self) -> FeatureId:
        return self.feature_id

    def get_child(self, branch_value: int) -> VertexId:
        return self.children[branch_value]

    def set_child(self, branch_value: int, child: VertexId) -> None:
        self.children[int(branch_value)] = child

    def get_children(self) -> Dict[int, VertexId]:
        return dict(self.children)


@dataclass
class MDDLeafVertex(MDDVertex):
    """Terminal MDD vertex labelled by a Boolean constant or a class label."""

    label: LeafLabel = TRUE_TAG

    def __post_init__(self) -> None:
        self.label = _normalize_leaf_label(self.label)

    def get_tag(self) -> LeafLabel:
        return self.label


class MultiValuedDecisionDiagram:
    """
    Multi-valued Decision Diagram (MDD).
    """

    true_tag = TRUE_TAG
    false_tag = FALSE_TAG

    def __init__(
        self,
        feature_names: Optional[Sequence[str]] = None,
        domain_sizes: Optional[Union[int, Sequence[int], Dict[FeatureId, int]]] = None,
    ) -> None:

        self.vertices: Dict[VertexId, Union[MDDDecisionVertex, MDDLeafVertex]] = {}
        self.root: Optional[VertexId] = None
         
        # feature_names is only for visualization. 
        self.feature_names = list(feature_names or []) 
        self.domain_sizes = domain_sizes    

        self.feature_ordering: Optional[List[FeatureId]] = []

    def __len__(self) -> int:
        return len(self.vertices)

    def get_vertices_cnt(self) -> int:
        return len(self.vertices)          

    def get_ith_vertex(self, ith: VertexId) -> Union[MDDDecisionVertex, MDDLeafVertex]:
        return self.vertices[ith]

    def is_leaf(self, obj: Union[VertexId, MDDVertex]) -> bool:
        vertex = obj
        if not isinstance(obj, MDDVertex):
            vertex = self.vertices[obj]
        return isinstance(vertex, MDDLeafVertex)

    def add_leaf(self, vertex_id: VertexId, label: LeafLabel, number: Optional[Any] = None) -> MDDLeafVertex:
        if vertex_id in self.vertices:
            raise ValueError(f"Duplicate vertex id: {vertex_id}")
        vertex = MDDLeafVertex(vertex_id=vertex_id, label=label, number=number)
        self.vertices[vertex_id] = vertex
        return vertex

    def add_decision_vertex(
        self,
        vertex_id: VertexId,
        feature_id: FeatureId,
        children: Optional[Dict[int, VertexId]] = None,
        number: Optional[Any] = None,
    ) -> MDDDecisionVertex:
        if vertex_id in self.vertices:
            raise ValueError(f"Duplicate vertex id: {vertex_id}")
        vertex = MDDDecisionVertex(
            vertex_id=vertex_id,
            feature_id=feature_id,
            children={int(k): v for k, v in (children or {}).items()},
            number=number,
        )
        self.vertices[vertex_id] = vertex
        return vertex

    def set_root(self, vertex_id: VertexId) -> None:
        if vertex_id not in self.vertices:
            raise ValueError(f"Unknown root vertex: {vertex_id}")
        self.root = vertex_id

    def get_domain_size(self, feature_id: FeatureId) -> int:
        """
        Return |Dom_f| for feature f.
        """
        if isinstance(self.domain_sizes, int):
            return self.domain_sizes
        if isinstance(self.domain_sizes, dict):
            return int(self.domain_sizes[feature_id])
        if isinstance(self.domain_sizes, Sequence):
            return int(self.domain_sizes[int(feature_id)])
        vertex_domain = [
            max(vertex.children.keys()) + 1
            for vertex in self.vertices.values()
            if isinstance(vertex, MDDDecisionVertex) and vertex.feature_id == feature_id and vertex.children
        ]
        if not vertex_domain:
            raise ValueError(f"Cannot infer domain size for feature {feature_id}")
        return max(vertex_domain)

    def edge_relations(self) -> Dict[int, Set[Tuple[VertexId, VertexId]]]:
        """Return the family E = {E_b}: branch label -> {(src, dst), ...}."""
        relations: Dict[int, Set[Tuple[VertexId, VertexId]]] = {}
        for src, vertex in self.vertices.items():
            if isinstance(vertex, MDDDecisionVertex):
                for branch_value, dst in vertex.children.items():
                    relations.setdefault(branch_value, set()).add((src, dst))
        return relations

    def validate(self) -> None:
        if self.root is None:
            raise ValueError("MDD root is not set")
        if self.root not in self.vertices:
            raise ValueError(f"Root {self.root} is not a vertex")
        
        if self.get_vertices_cnt() != self.actual_size():
            raise ValueError(f"Not all vertices are reachable from the root. Only {self.actual_size()} of {self.get_vertices_cnt()} are reachable")

        for vertex_id, vertex in self.vertices.items():
            if isinstance(vertex, MDDLeafVertex):
                continue

            domain = set(range(self.get_domain_size(vertex.feature_id)))
            actual = set(vertex.children.keys())
            if actual != domain:
                raise ValueError(
                    f"Vertex {vertex_id} of feature {vertex.feature_id} has branches "
                    f"{sorted(actual)}, expected {sorted(domain)}"
                )
            for child_id in vertex.children.values():
                if child_id not in self.vertices:
                    raise ValueError(f"Vertex {vertex_id} points to unknown child {child_id}")

        self._validate_acyclic_from_root()
        self._refresh_feature_ordering()

    def _validate_acyclic_from_root(self) -> None:
        visiting: Set[VertexId] = set()
        visited: Set[VertexId] = set()

        def dfs(vertex_id: VertexId) -> None:
            if vertex_id in visited:
                return
            if vertex_id in visiting:
                raise ValueError(f"MDD contains a cycle at vertex {vertex_id}")
            visiting.add(vertex_id)
            vertex = self.vertices[vertex_id]
            if isinstance(vertex, MDDDecisionVertex):
                for child_id in vertex.children.values():
                    dfs(child_id)
            visiting.remove(vertex_id)
            visited.add(vertex_id)

        dfs(self.root)

    def _refresh_feature_ordering(self) -> None:
        preferred_ordering = (
            list(self.feature_ordering)
            if isinstance(self.feature_ordering, list) and self.feature_ordering
            else None
        )
        features: List[FeatureId] = []
        feature_set: Set[FeatureId] = set()
        edges: Dict[FeatureId, Set[FeatureId]] = {} # construct the topology of the features used for the current MDD representation
        indegree: Dict[FeatureId, int] = {}

        def add_feature(feature_id: FeatureId) -> None:
            if feature_id in feature_set:
                return
            feature_set.add(feature_id)
            features.append(feature_id)
            edges.setdefault(feature_id, set())
            indegree.setdefault(feature_id, 0)

        queue: List[VertexId] = [self.root] if self.root is not None else []
        visited: Set[VertexId] = set()
        cursor = 0
        while cursor < len(queue):
            vertex_id = queue[cursor]
            cursor += 1
            if vertex_id in visited:
                continue
            visited.add(vertex_id)

            vertex = self.vertices[vertex_id]
            if isinstance(vertex, MDDLeafVertex):
                continue

            src_feature = vertex.feature_id
            add_feature(src_feature)
            for child_id in vertex.children.values():
                child = self.vertices[child_id]
                if isinstance(child, MDDDecisionVertex):
                    dst_feature = child.feature_id
                    add_feature(dst_feature)
                    if dst_feature == src_feature:
                        self.feature_ordering = None
                        return
                    if dst_feature not in edges[src_feature]:
                        edges[src_feature].add(dst_feature)
                        indegree[dst_feature] += 1
                queue.append(child_id)

        if preferred_ordering is not None and feature_set.issubset(set(preferred_ordering)):
            rank = {feature_id: idx for idx, feature_id in enumerate(preferred_ordering)}
            if all(rank[src] < rank[dst] for src, dsts in edges.items() for dst in dsts):   # check if the ordering property is satisfied
                self.feature_ordering = [feature_id for feature_id in preferred_ordering if feature_id in feature_set]
                return

        self.feature_ordering = self._topological_feature_ordering(features, edges, indegree)   

    @staticmethod
    def _topological_feature_ordering(
        features: List[FeatureId],
        edges: Dict[FeatureId, Set[FeatureId]],
        indegree: Dict[FeatureId, int],
    ) -> Optional[List[FeatureId]]:
        ordering: List[FeatureId] = []
        remaining = set(features)

        while remaining:
            zero_indegree = [feature_id for feature_id in features if feature_id in remaining and indegree[feature_id] == 0]
            if not zero_indegree:
                return None

            feature_id = zero_indegree[0]
            ordering.append(feature_id)
            remaining.remove(feature_id)
            for dst in edges[feature_id]:
                indegree[dst] -= 1

        return ordering

    def get_feature_ordering(self) -> Tuple[Optional[List[FeatureId]], Optional[List[str]]]:
        if self.feature_ordering is None:
            return None, None
        names = [self._feature_name(feature_id) for feature_id in self.feature_ordering]
        return list(self.feature_ordering), names

    def is_ordered(self) -> bool:
        self._refresh_feature_ordering()
        return self.feature_ordering is not None

    def evaluate(self, assignment: Union[Sequence[int], Dict[FeatureId, int]]) -> LeafLabel:
        if self.root is None:
            raise ValueError("MDD root is not set")

        vertex_id = self.root
        while True:
            vertex = self.vertices[vertex_id]
            if isinstance(vertex, MDDLeafVertex):
                return vertex.label
            if isinstance(assignment, dict):
                branch_value = int(assignment[vertex.feature_id])
            else:
                branch_value = int(assignment[int(vertex.feature_id)])
            vertex_id = vertex.children[branch_value]

    def predict_one(self, assignment: Union[Sequence[int], Dict[FeatureId, int]]) -> int:
        label = self.evaluate(assignment)
        if label == self.true_tag:
            return 1
        if label == self.false_tag:
            return 0
        return int(label)


    def predict(self, data: Iterable[Union[Sequence[int], Dict[FeatureId, int]]]) -> List[int]:
        return [self.predict_one(row) for row in data]

    def build(
        self,
        mdd_nodes: Sequence[VertexId],
        node_labels: Dict[VertexId, Union[FeatureId, LeafLabel]],
        edges: Union[Dict[int, Dict[VertexId, VertexId]], Dict[VertexId, Dict[int, VertexId]]],
        root: Optional[VertexId] = None,
        edge_format: str = "by_branch",
        prune_unreachable: bool = True,
        validate: bool = True,
    ) -> None:
        self.vertices.clear()
        node_set = set(mdd_nodes)
        for node in mdd_nodes:
            if node not in node_labels:
                raise ValueError(f"Missing label for vertex {node}")
            label = node_labels[node]
            if _is_boolean_leaf_label(label):
                self.add_leaf(node, label, number=node)
            else:
                self.add_decision_vertex(node, label, number=node)

        by_vertex = self._edges_to_by_vertex(edges, edge_format=edge_format)
        for src, children in by_vertex.items():
            if src not in node_set:
                raise ValueError(f"Unknown edge source {src}")
            vertex = self.vertices[src]
            if isinstance(vertex, MDDLeafVertex):
                raise ValueError(f"Leaf vertex {src} cannot have outgoing edges")
            for branch_value, dst in children.items():
                if dst not in node_set:
                    raise ValueError(f"Unknown edge target {dst}")
                vertex.set_child(branch_value, dst)

        self.root = mdd_nodes[0] if root is None else root

        if prune_unreachable:
            self.prune_unreachable()
        if validate:
            self.validate()

    @staticmethod
    def _edges_to_by_vertex(
        edges: Union[Dict[int, Dict[VertexId, VertexId]], Dict[VertexId, Dict[int, VertexId]]],
        edge_format: str,
    ) -> Dict[VertexId, Dict[int, VertexId]]:
        """ Format the input `edges` in the standard form: `by_vertex`, as it is more convenience for building MDD """
        if edge_format == "by_vertex":
            return {src: {int(b): dst for b, dst in children.items()} for src, children in edges.items()}
        if edge_format != "by_branch":
            raise ValueError("edge_format must be either 'by_branch' or 'by_vertex'")

        by_vertex: Dict[VertexId, Dict[int, VertexId]] = {}
        for branch_value, relation in edges.items():
            for src, dst in relation.items():
                by_vertex.setdefault(src, {})[int(branch_value)] = dst
        return by_vertex

    def to_dict(self) -> Dict[str, Any]:
        """Export a simple serializable representation for debugging/recovery."""
        vertices: Dict[str, Any] = {}
        for vertex_id, vertex in self.vertices.items():
            key = str(vertex_id)
            if isinstance(vertex, MDDLeafVertex):
                vertices[key] = {"type": "leaf", "label": vertex.label, "number": vertex.number}
            else:
                vertices[key] = {
                    "type": "decision",
                    "feature_id": vertex.feature_id,
                    "children": {str(k): v for k, v in sorted(vertex.children.items())},
                    "number": vertex.number,
                }
        return {
            "root": self.root,
            "vertices": vertices,
            "edge_relations": {
                str(branch): sorted(list(relation), key=lambda edge: str(edge))
                for branch, relation in self.edge_relations().items()
            },
            "feature_ordering": self.feature_ordering,
            "feature_names": self.feature_names,
        }

    def get_dot_description(
        self,
        branch_labels_by_feature: Optional[Dict[FeatureId, Union[List[str], Dict[int, str]]]] = None,
        edge_label_mode: str = "value",
        show_legend: bool = False,
    ) -> str:
        """Return a Graphviz DOT string.
        """
        if edge_label_mode not in ("value", "bin"):
            raise ValueError("edge_label_mode must be either 'value' or 'bin'")

        nodes_dot_info = ""
        edges_dot_info = ""

        for vertex_id, vertex in self.vertices.items():
            dot_id = self._dot_id(vertex_id)
            display_id = vertex.number if vertex.number is not None else vertex_id
            if isinstance(vertex, MDDLeafVertex):
                nodes_dot_info += f'\t{dot_id}[label="{vertex.label}", xlabel="n{display_id}", shape=square]\n'
                continue

            feature = self._feature_name(vertex.feature_id)
            nodes_dot_info += f'\t{dot_id}[label="{feature}", xlabel="n{display_id}"]\n'
            for branch_value, child_id in sorted(vertex.children.items()):
                edge_label = self._branch_label(
                    vertex.feature_id,
                    branch_value,
                    branch_labels_by_feature,
                    edge_label_mode=edge_label_mode,
                )
                edges_dot_info += (
                    f'\t{dot_id} -> {self._dot_id(child_id)}[label="{self._dot_quote_escape(edge_label)}"];\n'
                )

        ordering, names = self.get_feature_ordering()
        if ordering is None:
            ordering_info = '\n\tlabel="The MDD does not satisfy a level-wise ordering property!"'
        else:
            ordering_info = (
                f'\n\tlabel="The feature ordering: {ordering}'
                f'\\nThe corresponding feature name: [{",".join(names)}]"'
            )
        legend_info = self._dot_legend(branch_labels_by_feature) if show_legend else ""
        return "digraph G {\n" + nodes_dot_info + edges_dot_info + legend_info + ordering_info + "\n}"

    def reachable_vertices(self) -> Set[VertexId]:
        """Return vertices reachable from root; useful after merging/recovery."""
        if self.root is None:
            return set()
        reached: Set[VertexId] = set()
        stack = [self.root]
        while stack:
            vertex_id = stack.pop()
            if vertex_id in reached:
                continue
            reached.add(vertex_id)
            vertex = self.vertices[vertex_id]
            if isinstance(vertex, MDDDecisionVertex):
                stack.extend(vertex.children.values())
        return reached

    def prune_unreachable(self) -> None:
        """Remove vertices and edges that are not reachable from root."""
        reached = self.reachable_vertices()
        self.vertices = {
            vertex_id: vertex
            for vertex_id, vertex in self.vertices.items()
            if vertex_id in reached
        }
        # self.validate()

    def actual_size(self) -> int:
        """Number of vertices in the explicit MDD reachable from the root."""
        return len(self.reachable_vertices())


    def _feature_name(self, feature_id: FeatureId) -> str:
        if isinstance(feature_id, int) and 0 <= feature_id < len(self.feature_names):
            return self.feature_names[feature_id]
        return str(feature_id)

    def _branch_label(
        self,
        feature_id: FeatureId,
        branch_value: int,
        branch_labels_by_feature: Optional[Dict[FeatureId, Union[List[str], Dict[int, str]]]],
        edge_label_mode: str,
    ) -> str:
        if edge_label_mode == "value" or branch_labels_by_feature is None:
            return str(branch_value)
        labels = branch_labels_by_feature.get(feature_id)
        if labels is None:
            return str(branch_value)
        if isinstance(labels, dict):
            return str(labels.get(branch_value, branch_value))
        if 0 <= branch_value < len(labels):
            return str(labels[branch_value])
        return str(branch_value)

    def _dot_legend(
        self,
        branch_labels_by_feature: Optional[Dict[FeatureId, Union[List[str], Dict[int, str]]]],
    ) -> str:
        used_features = []
        seen = set()
        for vertex in self.vertices.values():
            if isinstance(vertex, MDDDecisionVertex) and vertex.feature_id not in seen:
                seen.add(vertex.feature_id)
                used_features.append(vertex.feature_id)

        if not used_features:
            return ""

        rows = [
            '<TR><TD BGCOLOR="lightgray"><B>Feature</B></TD>'
            '<TD BGCOLOR="lightgray"><B>Bins</B></TD></TR>'
        ]
        for feature_id in used_features:
            feature_name = self._dot_html_escape(self._feature_name(feature_id))
            bin_text = ""
            if branch_labels_by_feature is not None and feature_id in branch_labels_by_feature:
                labels = branch_labels_by_feature[feature_id]
                if isinstance(labels, dict):
                    items = sorted(labels.items())
                else:
                    items = list(enumerate(labels))
                bin_text = "<BR/>".join(
                    f"{idx}: {self._dot_html_escape(str(label))}"
                    for idx, label in items
                )
            rows.append(f"<TR><TD>{feature_id}: {feature_name}</TD><TD ALIGN=\"LEFT\">{bin_text}</TD></TR>")

        table = "".join(rows)
        return (
            '\tlegend[shape=plain, label=<\n'
            f'\t<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0">{table}</TABLE>\n'
            '\t>]\n'
        )

    @staticmethod
    def _dot_id(vertex_id: VertexId) -> str:
        text = str(vertex_id).replace(" ", "_").replace("-", "_")
        text = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
        return f"n{text}"

    @staticmethod
    def _dot_html_escape(text: str) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    @staticmethod
    def _dot_quote_escape(text: str) -> str:
        return str(text).replace("\\", "\\\\").replace('"', r"\"").replace("\n", r"\n")


def _to_nested_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _argmax(values: Sequence[float]) -> int:
    if len(values) == 0:
        raise ValueError("Cannot argmax over an empty sequence")
    return max(range(len(values)), key=lambda idx: values[idx])


def _is_all_zero(values: Sequence[float]) -> bool:
    return all(float(value) == 0.0 for value in values)


# Short aliases used by downstream code.
MDD = MultiValuedDecisionDiagram
DecisionVertex = MDDDecisionVertex
LeafVertex = MDDLeafVertex


if __name__ == '__main__':
    pass
    mdd_nodes = ["0", "1"]
    node_labels = {"0" : 0, "1" :"T"}
    edges = {
        0 : {"0" : "1"},
        1 : {"0" : "1"},
        2 : {"0" : "1"},
    }
    root = "0"

    mdd = MDD()
    mdd.build(
        mdd_nodes=mdd_nodes,
        node_labels=node_labels,
        edges=edges,
        root=root,
        edge_format="by_branch"
    )

    dot = mdd.get_dot_description()
    print("dot: ", dot)
