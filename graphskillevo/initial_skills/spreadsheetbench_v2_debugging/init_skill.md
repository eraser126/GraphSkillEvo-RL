# SpreadsheetBench 2 Debugging Skill

## Global Guidance

- Audit the supplied workbook against the user's requested scope. Read `INPUT_PATH` and save the repaired workbook to `OUTPUT_PATH`.
- Use `openpyxl` to inspect and edit the workbook. Preserve sheet layout, existing formulas, and formatting outside the intended repair. A cached value is an observation about a prior calculation, not a substitute for reading the formula.
- Select a workflow by the requested scope. When checkpoint tooling is available, record each node after completing its work and cite the relevant sheet names, cell addresses, or tool observations.

## Node Lists

### Map Workbook and Scope
- List the worksheets and identify their roles from headers, labels, and formula references. Record the repair scope and any explicit instructions about what must stay unchanged.

### Trace Formula Anomalies
- Compare suspicious formulas with nearby equivalent formulas and follow upstream references across sheets. Check ranges, anchors, lookup keys, signs, units, and duplicate inputs where relevant. Separate likely causes from downstream symptoms.

### Compare Formatting Conventions
- Compare analogous inputs and formulas within the same workbook. Use nearby cells serving the same role to infer the intended style. Record the inconsistent style property and the cells that support the inference.

### Plan Minimal Repairs
- List the intended edits and the evidence supporting each. Consider which downstream outputs may legitimately change. Leave uncertain or unrelated cells untouched. Keep cell contents intact for style-only changes.

### Write Repair Script
- Write `solution.py` using `INPUT_PATH` and `OUTPUT_PATH`. Apply only the planned edits to a loaded workbook, then save to `OUTPUT_PATH`. Use a narrow style edit when a full style copy would change unrelated properties.

### Execute and Inspect Output
- Run `solution.py`, check its exit status, and reopen the output workbook. Inspect the edited cells and a sample of affected downstream formulas. If a recalculation tool is available and the repair affects computed values, recalculate the output before checking them.

### Verify Repair and Scope
- Compare input and output to confirm the intended cell or style changes. Check the original anomaly, relevant downstream effects, and representative cells that should stay unchanged. Revisit diagnosis or implementation if these checks fail. Submit the verified output workbook.

## Task Graphs

### Formula and Reference Debugging
**Use when:** The task asks for an audit, calculation repair, formula repair, or reference correction.

**Workflow:**
1. Map Workbook and Scope
2. Trace Formula Anomalies
3. Plan Minimal Repairs
4. Write Repair Script
5. Execute and Inspect Output
6. Verify Repair and Scope

### Formatting Consistency Repair
**Use when:** The task specifically asks to fix formatting or color conventions.

**Workflow:**
1. Map Workbook and Scope
2. Compare Formatting Conventions
3. Plan Minimal Repairs
4. Write Repair Script
5. Execute and Inspect Output
6. Verify Repair and Scope

### Combined Audit
**Use when:** The task explicitly asks to review both calculation logic and formatting.

**Workflow:**
1. Map Workbook and Scope
2. Trace Formula Anomalies
3. Compare Formatting Conventions
4. Plan Minimal Repairs
5. Write Repair Script
6. Execute and Inspect Output
7. Verify Repair and Scope
