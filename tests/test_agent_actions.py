"""Tests for Phase 5.3 agent action registry."""

from __future__ import annotations

from gpuboost.agent.actions import (
    ACTION_REGISTRY,
    ANALYZE_CODE,
    CREATE_PATCH_PLAN,
    GENERATE_DIFF,
    GENERATE_RECOMMENDATIONS,
    INSPECT_SYSTEM,
    RUN_QUICK_BENCHMARK,
    SUMMARIZE_RESULTS,
    ActionDefinition,
    create_agent_action,
    get_action_definition,
    is_known_action,
    list_action_definitions,
)


EXPECTED_ACTIONS = [
    INSPECT_SYSTEM,
    RUN_QUICK_BENCHMARK,
    GENERATE_RECOMMENDATIONS,
    ANALYZE_CODE,
    CREATE_PATCH_PLAN,
    GENERATE_DIFF,
    SUMMARIZE_RESULTS,
]


def test_registry_contains_all_expected_actions() -> None:
    assert list(ACTION_REGISTRY) == EXPECTED_ACTIONS
    assert set(ACTION_REGISTRY) == set(EXPECTED_ACTIONS)


def test_list_action_definitions_returns_deterministic_order() -> None:
    definitions = list_action_definitions()

    assert [definition.name for definition in definitions] == EXPECTED_ACTIONS


def test_get_action_definition_returns_expected_metadata() -> None:
    definition = get_action_definition(RUN_QUICK_BENCHMARK)

    assert definition is not None
    assert definition.name == RUN_QUICK_BENCHMARK
    assert definition.required is True
    assert definition.produces == ["benchmark_result"]
    assert definition.requires == ["gpu_profile"]
    assert definition.safe_by_default is True


def test_get_action_definition_returns_none_for_unknown() -> None:
    assert get_action_definition("unknown_action") is None


def test_is_known_action_true_false() -> None:
    assert is_known_action(INSPECT_SYSTEM) is True
    assert is_known_action("unknown_action") is False


def test_create_agent_action_for_known_action_populates_description_and_required() -> None:
    action = create_agent_action(GENERATE_RECOMMENDATIONS)
    definition = get_action_definition(GENERATE_RECOMMENDATIONS)

    assert definition is not None
    assert action.id == GENERATE_RECOMMENDATIONS
    assert action.name == GENERATE_RECOMMENDATIONS
    assert action.description == definition.description
    assert action.required == definition.required
    assert action.depends_on == []
    assert action.inputs == {}
    assert action.status == "pending"
    assert action.error is None


def test_create_agent_action_supports_custom_id() -> None:
    action = create_agent_action(ANALYZE_CODE, action_id="action_004")

    assert action.id == "action_004"
    assert action.name == ANALYZE_CODE


def test_create_agent_action_supports_inputs_and_depends_on() -> None:
    action = create_agent_action(
        CREATE_PATCH_PLAN,
        inputs={"script_path": "train.py", "quick": True},
        depends_on=["analyze_code"],
    )

    assert action.inputs == {"script_path": "train.py", "quick": True}
    assert action.depends_on == ["analyze_code"]


def test_create_agent_action_handles_unknown_action() -> None:
    action = create_agent_action("unknown_action")

    assert action.id == "unknown_action"
    assert action.name == "unknown_action"
    assert action.description == "Unknown action: unknown_action"
    assert action.required is False
    assert action.depends_on == []
    assert action.inputs == {}
    assert action.status == "pending"


def test_action_definition_list_defaults_are_isolated() -> None:
    first = ActionDefinition(
        name="first",
        description="First action.",
        required=True,
    )
    second = ActionDefinition(
        name="second",
        description="Second action.",
        required=False,
    )

    first.produces.append("first_artifact")
    first.requires.append("first_input")

    assert first.produces == ["first_artifact"]
    assert first.requires == ["first_input"]
    assert second.produces == []
    assert second.requires == []
