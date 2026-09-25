from __future__ import annotations

import pytest

from shared.query_planner import (
    extract_entity_id,
    plan_inventory_supplier_queries,
)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Qual é a política do M10?", "M10"),
        ("Compare o fornecedor do a100 com o estoque.", "A100"),
        ("O item B-200 pode ser reposto?", "B-200"),
        ("Explique a política de estoque.", None),
    ],
)
def test_extract_entity_id(question, expected):
    assert extract_entity_id(question) == expected


def test_plan_creates_domain_specific_queries():
    plan = plan_inventory_supplier_queries(
        "O fornecedor do M10 consegue atender a política de estoque atual?"
    )

    assert plan.entity_id == "M10"
    assert "M10" in plan.inventory
    assert "Retrieval V2" in plan.inventory
    assert "target stock level" in plan.inventory
    assert "supplier name" in plan.supplier
    assert "contractual lead time" in plan.supplier
    assert "original request" in plan.inventory.lower()


def test_plan_without_entity_keeps_safe_fallback():
    plan = plan_inventory_supplier_queries(
        "O fornecedor consegue atender a política de estoque?"
    )

    assert plan.entity_id is None
    assert "item mentioned in the original request" in plan.inventory
    assert "item mentioned in the original request" in plan.supplier


def test_plan_rejects_empty_input():
    with pytest.raises(ValueError, match="user_input cannot be empty"):
        plan_inventory_supplier_queries("   ")


def test_plan_three_specialists_creates_logistics_query_without_product_code():
    from shared.query_planner import plan_specialist_queries

    plan = plan_specialist_queries(
        "Crie um plano de reposição do M10 considerando estoque, fornecedor e transporte.",
        ("inventory", "supplier", "logistics"),
    )

    assert plan.entity_id == "M10"
    assert plan.inventory is not None and "M10" in plan.inventory
    assert plan.supplier is not None and "M10" in plan.supplier
    assert plan.logistics is not None
    assert "M10" in plan.logistics
    assert "transportation modes" in plan.logistics
    assert set(plan.as_dict()) == {"inventory", "supplier", "logistics"}


def test_plan_only_requested_specialists():
    from shared.query_planner import plan_specialist_queries

    plan = plan_specialist_queries(
        "Compare o fornecedor e o transporte do A100.",
        ("supplier", "logistics"),
    )

    assert plan.inventory is None
    assert plan.supplier is not None
    assert plan.logistics is not None
    assert set(plan.as_dict()) == {"supplier", "logistics"}


def test_plan_rejects_unknown_specialist():
    from shared.query_planner import plan_specialist_queries

    with pytest.raises(ValueError, match="Unsupported specialist"):
        plan_specialist_queries("Pergunta sobre M10", ("finance",))
