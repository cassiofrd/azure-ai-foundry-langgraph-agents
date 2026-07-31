from shared.response_parser import extract_evidence


def test_extracts_inline_evidence_and_deduplicates():
    answer = """
Evidence:
- Title: Bolt M10 inventory policy, Source: manual, Entity_id: M10
- Title: Bolt M10 inventory policy, Source: manual, Entity_id: M10
"""
    evidence = extract_evidence(answer)
    assert len(evidence) == 1
    assert evidence[0].title == "Bolt M10 inventory policy"
    assert evidence[0].source == "manual"
    assert evidence[0].entity_id == "M10"


def test_extracts_multiline_evidence():
    answer = """
Evidence:
- Title: Bolt M10 approved supplier
- Source: supplier_master_data
- Entity_id: M10
"""
    evidence = extract_evidence(answer)
    assert len(evidence) == 1
    assert evidence[0].source == "supplier_master_data"


def test_empty_answer_has_no_evidence():
    assert extract_evidence("  ") == ()
