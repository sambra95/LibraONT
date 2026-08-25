# Workflow schematic

Two figures, each a standalone script so it can be adjusted on its own.

| file | figure |
| --- | --- |
| `panel_sankey.py` | `workflow.svg` - a Sankey of reads flowing through the filters, losses peeling away, and the outputs each surviving pool feeds |
| `panel_criteria.py` | `criteria.svg` - one diagram per classification rule |
| `build.py` | renders both |
| `svgkit.py` | SVG primitives; colours and card styling come from the app |
| `steps.py` | the filter sequence and its illustrative proportions |

```
python docs/workflow/build.py            # both figures
python docs/workflow/panel_sankey.py     # just workflow.svg
```

The Sankey carries no counts on purpose - the `keep` fractions in `steps.py`
are illustrative, so the figure describes the workflow rather than one run.
Live counts appear in the app's *Read filtering, step by step* table.

`app.py` embeds both files in an expander at the top of the page, so rebuilding
here updates the app.
