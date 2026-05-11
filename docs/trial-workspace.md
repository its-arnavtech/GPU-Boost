# Trial Workspace

Phase 7 adds safe trial validation for generated patch suggestions:

```bash
gpuboost agent optimize train.py --trial
gpuboost agent optimize train.py --trial --json
gpuboost agent optimize train.py --trial --test "pytest"
```

Trial mode creates a temporary workspace, copies the target file into it, and
applies generated patches only to the copy. The original source file is never
modified, and GPUBoost still has no `--apply` command.

Syntax validation uses Python compilation only. It does not import or execute
the target script.

The `--test` command is opt-in. It may execute arbitrary user-provided code and
only runs when passed together with `--trial`.

Phase 7 does not add history storage, model/data pipeline features, or LLM
logic.
