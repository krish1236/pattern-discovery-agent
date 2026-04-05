"""Research router query generation."""

from src.packs.research.router import _generate_queries, build_source_plan


def test_generate_queries_splits_topic_and_dedupes() -> None:
    topic = "Bitcoin layer-2 scaling Lightning rollups"
    qs = _generate_queries(topic)
    lowered = [q.lower() for q in qs]
    assert topic.lower() in lowered
    assert "bitcoin layer-2" in lowered
    assert "layer-2 scaling" in lowered
    assert len(qs) == len(set(lowered))


def test_build_source_plan_includes_subqueries() -> None:
    plan = build_source_plan("Bitcoin layer-2 scaling", focus="all")
    assert len(plan.queries) >= 4
    assert any("bitcoin" in q.lower() for q in plan.queries)
