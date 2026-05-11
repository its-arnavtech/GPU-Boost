"""Known deterministic agent action definitions for GPUBoost."""

from __future__ import annotations

from dataclasses import dataclass, field

from gpuboost.schemas.agent import AgentAction, AgentValue


INSPECT_SYSTEM = "inspect_system"
RUN_QUICK_BENCHMARK = "run_quick_benchmark"
GENERATE_RECOMMENDATIONS = "generate_recommendations"
ANALYZE_CODE = "analyze_code"
CREATE_PATCH_PLAN = "create_patch_plan"
GENERATE_DIFF = "generate_diff"
RUN_TRIAL_WORKSPACE = "run_trial_workspace"
SUMMARIZE_RESULTS = "summarize_results"

ACTION_ORDER = (
    INSPECT_SYSTEM,
    RUN_QUICK_BENCHMARK,
    GENERATE_RECOMMENDATIONS,
    ANALYZE_CODE,
    CREATE_PATCH_PLAN,
    GENERATE_DIFF,
    RUN_TRIAL_WORKSPACE,
    SUMMARIZE_RESULTS,
)


@dataclass(slots=True)
class ActionDefinition:
    """Metadata for a known deterministic agent action."""

    name: str
    description: str
    required: bool
    produces: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    safe_by_default: bool = True


ACTION_REGISTRY: dict[str, ActionDefinition] = {
    INSPECT_SYSTEM: ActionDefinition(
        name=INSPECT_SYSTEM,
        description="Inspect the available GPU and software environment.",
        required=True,
        produces=["gpu_profile"],
        requires=[],
        safe_by_default=True,
    ),
    RUN_QUICK_BENCHMARK: ActionDefinition(
        name=RUN_QUICK_BENCHMARK,
        description="Run the quick benchmark suite.",
        required=True,
        produces=["benchmark_result"],
        requires=["gpu_profile"],
        safe_by_default=True,
    ),
    GENERATE_RECOMMENDATIONS: ActionDefinition(
        name=GENERATE_RECOMMENDATIONS,
        description="Generate deterministic optimization recommendations.",
        required=True,
        produces=["advisor_result"],
        requires=["benchmark_result"],
        safe_by_default=True,
    ),
    ANALYZE_CODE: ActionDefinition(
        name=ANALYZE_CODE,
        description="Analyze the target script for GPU optimization opportunities.",
        required=False,
        produces=["code_analysis"],
        requires=["script_path"],
        safe_by_default=True,
    ),
    CREATE_PATCH_PLAN: ActionDefinition(
        name=CREATE_PATCH_PLAN,
        description="Create a reviewable patch plan from code analysis findings.",
        required=False,
        produces=["patch_plan"],
        requires=["code_analysis"],
        safe_by_default=True,
    ),
    GENERATE_DIFF: ActionDefinition(
        name=GENERATE_DIFF,
        description="Generate a reviewable diff from a patch plan.",
        required=False,
        produces=["diff"],
        requires=["patch_plan"],
        safe_by_default=True,
    ),
    RUN_TRIAL_WORKSPACE: ActionDefinition(
        name=RUN_TRIAL_WORKSPACE,
        description=(
            "Validate generated patch suggestions in a temporary trial workspace."
        ),
        required=False,
        produces=["trial_result"],
        requires=["patch_plan"],
        safe_by_default=True,
    ),
    SUMMARIZE_RESULTS: ActionDefinition(
        name=SUMMARIZE_RESULTS,
        description="Summarize the agent run results.",
        required=True,
        produces=["summary"],
        requires=[],
        safe_by_default=True,
    ),
}


def get_action_definition(name: str) -> ActionDefinition | None:
    """Return the registered action definition for a name, if known."""

    return ACTION_REGISTRY.get(name)


def is_known_action(name: str) -> bool:
    """Return whether an action name is registered."""

    return name in ACTION_REGISTRY


def list_action_definitions() -> list[ActionDefinition]:
    """Return registered action definitions in deterministic order."""

    return [ACTION_REGISTRY[name] for name in ACTION_ORDER]


def create_agent_action(
    name: str,
    action_id: str | None = None,
    inputs: dict[str, AgentValue] | None = None,
    depends_on: list[str] | None = None,
) -> AgentAction:
    """Create an AgentAction from registry metadata without executing it."""

    definition = get_action_definition(name)
    if definition is None:
        description = f"Unknown action: {name}"
        required = False
    else:
        description = definition.description
        required = definition.required

    return AgentAction(
        id=action_id or name,
        name=name,
        description=description,
        required=required,
        depends_on=depends_on or [],
        inputs=inputs or {},
    )
