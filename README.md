# visforge

VisForge generates and verifies visual question-answering data, then evaluates changes to the synthesis rules.

Install the project with `pip install -e '.[dev]'`. Run `pytest` for local tests and `visforge cycle --help` for the experiment interface.

Start another cycle in a new output directory with `--resume-from path/to/completed-run` to use that run's selected library and proposal history. A cycle pins the inherited history alongside its initial library. The completed run records the selected version and prior outcomes in `evolution-history.json`.

The observer can set the planner's selection instruction, add or edit templates, and register reusable compositions of existing operators. It can submit dependent changes together with a `batch` edit. Composition definitions declare typed input ports, textual parameters, steps, and a result, and are checked before generation. The planner sees the expanded operator chain, and question traces identify every composition it executed.

Set `VISFORGE_TARGET_MODEL=DeepSeek-V4.1-Flash` in the private environment file or pass `--target-model DeepSeek-V4.1-Flash` to use the campus gateway for target answers. The local Qwen checkpoint answers as the reference, and Luna audits questions and judges answer equivalence. Both solvers answer verified questions from the image and question alone. Each run pins its target assignment and records external target usage separately. Start a new output directory to change the target.
