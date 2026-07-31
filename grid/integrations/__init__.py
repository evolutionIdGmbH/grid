"""Client-side integrations: typed-pipeline frameworks that declare output
schemas before the request (docs/integrations-plan.md item 4).

Import the framework-specific module directly, e.g.
``from grid.integrations.dspy_adapter import GridJSONAdapter`` - each module
guards its framework import so grid itself never grows the dependency.
"""
