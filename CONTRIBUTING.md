# Contributing

This branch maintains V4. V4 tensor semantics are frozen; V5 is in development. Version changes are installed explicitly through separate releases, not selected by a compatibility switch. Preserve the public MHD_Node, MHD_Edge, MHD_Topo and MHD_Graph names.

Use lowercase snake_case modules under src/mhd_framework. Core graph semantics belong in core.py; optional data, training and distributed utilities in utils.py. Separate packaging maintenance from scientific behavior changes. Changes require source, tests and API documentation to agree.

Run `python scripts/manage.py test`, `python -m build`, and verify wheel imports outside the checkout. Distributed/GPU checks require explicit execution and hardware reporting; never generalize CPU validation into a multi-GPU claim. Keep personal paths, cluster accounts, datasets and application queues outside the toolbox.
