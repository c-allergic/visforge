# visforge

VisForge generates and verifies visual question-answering data, then evaluates changes to the synthesis rules.

Install the project with `pip install -e '.[dev]'`. Run `pytest` for local tests and `visforge cycle --help` for the experiment interface.

Start another cycle in a new output directory with `--resume-from path/to/completed-run` to use that run's selected library and proposal history. A cycle pins the inherited history alongside its initial library. The completed run records the selected version and prior outcomes in `evolution-history.json`.

The observer can set the planner's selection instruction, add or edit templates, and register reusable compositions of existing operators. It can submit dependent changes together with a `batch` edit. Composition definitions declare typed input ports, textual parameters, steps, and a result, and are checked before generation. The planner sees the expanded operator chain, and question traces identify every composition it executed.
