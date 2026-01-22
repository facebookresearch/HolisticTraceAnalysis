Optimized Critical Path Visualization
=====================================

A new visualization mode has been added to improve the clarity of critical path overlays.

This mode filters out non-essential intermediate nodes(except at the start and end of the path). 

This reduces clutter and helps focus on key performance events.

Usage
-----

Call the method :meth:`TraceAnalysis.overlay_critical_change_path_analysis` with appropriate arguments to generate the cleaner trace overlay.

This enhancement aids in debugging and performance analysis in complex heterogeneous workloads.

Example::

    analyzer.overlay_critical_change_path_analysis(
        rank=0,
        critical_path_graph=cp_graph,
        output_dir="path/to/output"
    )
