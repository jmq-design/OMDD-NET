"""
Bridge between the paper-style MDD in utils.MDD and dd.mdd.MDD.

- utils.MDD.MultiValuedDecisionDiagram stores the explicit tuple (V, root, var, E) used in the paper and has no complemented edges.
- dd.mdd.MDD is a canonical ordered MDD package representation whose edges can be complemented by using negative integer references.

"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import dd.mdd as _dd_mdd

from utils.MDD import (
    FALSE_TAG,
    TRUE_TAG,
    FeatureId,
    LeafLabel,
    MDD,
    MDDDecisionVertex,
    MDDLeafVertex,
    MultiValuedDecisionDiagram,
)


DDMDDRef = int
FeatureToVarName = Callable[[FeatureId], str]
VarNameToFeature = Callable[[str], FeatureId]


def default_feature_to_var_name(feature_id: FeatureId) -> str:
    """Default variable naming used when constructing dd.mdd.MDD."""
    if isinstance(feature_id, int):
        return f"f{feature_id}"
    return str(feature_id)


def default_var_name_to_feature(var_name: str) -> FeatureId:
    """Inverse of default_feature_to_var_name where possible."""
    if var_name.startswith("f") and var_name[1:].isdigit():
        return int(var_name[1:])
    return var_name


def MDD_to_mdd(
    explicit_mdd: MultiValuedDecisionDiagram,
    feature_to_var_name: Optional[FeatureToVarName] = None,
) -> Tuple[_dd_mdd.MDD, DDMDDRef, Dict[FeatureId, str]]:
    """
    Convert an explicit paper-style MDD to dd.mdd.MDD.
    """
    explicit_mdd.validate()
    if explicit_mdd.feature_ordering is None:
        raise ValueError("dd.mdd.MDD requires an ordered MDD; got feature_ordering=None")

    feature_to_var_name = feature_to_var_name or default_feature_to_var_name
    feature_ordering = list(explicit_mdd.feature_ordering)
    feature_to_var = {feature_id: feature_to_var_name(feature_id) for feature_id in feature_ordering}

    if len(set(feature_to_var.values())) != len(feature_to_var):
        raise ValueError(f"Non-unique dd.mdd variable names: {feature_to_var}")

    dvars = {
        feature_to_var[feature_id]: {
            "level": level,
            "len": explicit_mdd.get_domain_size(feature_id),
        }
        for level, feature_id in enumerate(feature_ordering)
    }
    dd_mdd = _dd_mdd.MDD(dvars)

    cache: Dict[Any, DDMDDRef] = {}

    def convert(vertex_id: Any) -> DDMDDRef:
        if vertex_id in cache:
            return cache[vertex_id]

        vertex = explicit_mdd.get_ith_vertex(vertex_id)
        if isinstance(vertex, MDDLeafVertex):
            root = _leaf_to_ref(vertex.label)
        elif isinstance(vertex, MDDDecisionVertex):
            var_name = feature_to_var[vertex.feature_id]
            level = dd_mdd.level_of_var(var_name)
            domain_size = explicit_mdd.get_domain_size(vertex.feature_id)
            children = tuple(convert(vertex.children[branch]) for branch in range(domain_size))
            root = dd_mdd.find_or_add(level, *children)
        else:
            raise TypeError(type(vertex))

        cache[vertex_id] = root
        return root

    root = convert(explicit_mdd.root)
    dd_mdd.incref(root)
    return dd_mdd, root, feature_to_var


def mdd_to_MDD(
    dd_mdd: _dd_mdd.MDD,
    root: DDMDDRef,
    var_name_to_feature: Optional[VarNameToFeature] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> MDD:
    """
    Recover a paper-style explicit MDD from dd.mdd.MDD and a root edge.
    """
    recover_dict, _, _ = mdd_recover(dd_mdd, root)
    return MDD_from_recover_dict(
        recover_dict,
        dd_mdd,
        var_name_to_feature=var_name_to_feature,
        feature_names=feature_names,
    )


def mdd_recover(
    dd_mdd: _dd_mdd.MDD,
    root: DDMDDRef,
) -> Tuple[Dict[Any, Any], int, set]:
    """
    Recover the reachable dd.mdd subgraph into a no-complement-edge dictionary.
    """
    if abs(root) not in dd_mdd:
        raise ValueError(f"Unknown dd.mdd root: {root}")

    recover_dict: Dict[Any, Any] = {
        "level_of_var": {var: desc["level"] for var, desc in dd_mdd.vars.items()},
        "roots": _recover_node_id(root),
    }
    to_visit = {root}
    visited = set()

    while to_visit:
        edge = to_visit.pop()
        recovered_id = _recover_node_id(edge)
        if recovered_id in visited:
            continue

        if abs(edge) == 1:
            recover_dict[recovered_id] = [recovered_id]
            visited.add(recovered_id)
            continue

        level, *children = dd_mdd._succ[abs(edge)]
        if edge < 0:
            children = [-child for child in children]

        recovered_children = [_recover_node_id(child) for child in children]
        recover_dict[recovered_id] = [level, *recovered_children]
        visited.add(recovered_id)
        to_visit.update(children)

    return recover_dict, len(visited), visited


def MDD_from_recover_dict(
    recover_dict: Dict[Any, Any],
    dd_mdd: _dd_mdd.MDD,
    var_name_to_feature: Optional[VarNameToFeature] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> MDD:
    """Build utils.MDD.MDD from the no-complement-edge dictionary."""
    var_name_to_feature = var_name_to_feature or default_var_name_to_feature
    level_to_var = {desc["level"]: var for var, desc in dd_mdd.vars.items()}
    domain_sizes_by_feature = {
        var_name_to_feature(var): desc["len"]
        for var, desc in dd_mdd.vars.items()
    }

    explicit_mdd = MDD(feature_names=feature_names, domain_sizes=domain_sizes_by_feature)

    for node_id, payload in recover_dict.items():
        if node_id in ("level_of_var", "roots"):
            continue
        if node_id == "T":
            explicit_mdd.add_leaf(node_id, TRUE_TAG)
        elif node_id == "F":
            explicit_mdd.add_leaf(node_id, FALSE_TAG)
        else:
            level = payload[0]
            var_name = level_to_var[level]
            explicit_mdd.add_decision_vertex(
                node_id,
                feature_id=var_name_to_feature(var_name),
                number=node_id,
            )

    for node_id, payload in recover_dict.items():
        if node_id in ("level_of_var", "roots", "T", "F"):
            continue
        vertex = explicit_mdd.get_ith_vertex(node_id)
        children = payload[1:]
        for branch_value, child_id in enumerate(children):
            vertex.set_child(branch_value, child_id)

    explicit_mdd.set_root(recover_dict["roots"])
    explicit_mdd.validate()
    return explicit_mdd


def mdd_actual_size(dd_mdd: _dd_mdd.MDD, root: DDMDDRef) -> int:
    """Return the no-complement-edge size of the dd.mdd rooted at root."""
    _, actual_size, _ = mdd_recover(dd_mdd, root)
    return actual_size


def mdd_predict_one(dd_mdd: _dd_mdd.MDD, root: DDMDDRef, assignment: Dict[str, int]) -> int:
    """
    Evaluate a dd.mdd root on one assignment.
    """
    edge = root
    complemented = False
    while abs(edge) != 1:
        if edge < 0:
            complemented = not complemented
        level, *children = dd_mdd._succ[abs(edge)]
        var = dd_mdd.var_at_level(level)
        value = int(assignment[var])
        edge = children[value]

    value = edge == 1
    if complemented:
        value = not value
    return int(value)


def mdd_predict(
    dd_mdd: _dd_mdd.MDD,
    root: DDMDDRef,
    data: Iterable[Union[Sequence[int], Dict[Union[str, FeatureId], int]]],
    feature_to_var: Optional[Dict[FeatureId, str]] = None,
) -> List[int]:

    return [
        mdd_predict_one(dd_mdd, root, _row_to_assignment(row, dd_mdd, feature_to_var))
        for row in data
    ]


def _row_to_assignment(
    row: Union[Sequence[int], Dict[Union[str, FeatureId], int]],
    dd_mdd: _dd_mdd.MDD,
    feature_to_var: Optional[Dict[FeatureId, str]],
) -> Dict[str, int]:
    if isinstance(row, dict):
        assignment: Dict[str, int] = {}
        for var_name in dd_mdd.vars:
            if var_name in row:
                assignment[var_name] = int(row[var_name])
                continue
            if feature_to_var is None:
                feature_id = default_var_name_to_feature(var_name)
            else:
                reverse = {var: feature_id for feature_id, var in feature_to_var.items()}
                feature_id = reverse[var_name]
            assignment[var_name] = int(row[feature_id])
        return assignment

    if feature_to_var is None:
        assignment = {}
        for level in range(len(dd_mdd.vars)):
            var_name = dd_mdd.var_at_level(level)
            feature_id = default_var_name_to_feature(var_name)
            if not isinstance(feature_id, int):
                feature_id = level
            assignment[var_name] = int(row[feature_id])
        return assignment

    return {var_name: int(row[feature_id]) for feature_id, var_name in feature_to_var.items()}


def _recover_node_id(edge: DDMDDRef) -> Union[int, str]:
    if edge == 1:
        return "T"
    if edge == -1:
        return "F"
    return edge


def _leaf_to_ref(label: LeafLabel) -> DDMDDRef:
    if label == TRUE_TAG or label is True or label == 1:
        return 1
    if label == FALSE_TAG or label is False or label == 0:
        return -1
    raise ValueError(f"dd.mdd.MDD represents Boolean functions; unsupported leaf label: {label}")


__all__ = [
    "MDD_to_mdd",
    "mdd_to_MDD",
    "mdd_recover",
    "MDD_from_recover_dict",
    "mdd_actual_size",
    "mdd_predict_one",
    "mdd_predict",
    "default_feature_to_var_name",
    "default_var_name_to_feature",
]
