# Demos Overview

See LongProbe in action with these live demonstrations showcasing key features and workflows.

## Available Demos

### 1. Complete Workflow
**Full RAG regression testing workflow**

Perfect for: Understanding the complete LongProbe workflow, first-time users, comprehensive testing

[View Demo →](complete-workflow.md){ .md-button .md-button--primary }

---

### 2. Monitor RAG Quality
**Detailed quality monitoring with Python API**

Perfect for: Production monitoring, detailed analysis, debugging

[View Demo →](monitor-quality.md){ .md-button .md-button--primary }

---

### 3. Detect Regressions
**Baseline comparison and regression detection**

Perfect for: CI/CD pipelines, deployment safety checks, version comparison

[View Demo →](detect-regressions.md){ .md-button .md-button--primary }

---

## Demo Features

All demos showcase:

- ✨ **Professional TUI** with Rich library
- 📊 **Live progress updates** showing real-time execution
- 🎨 **Color-coded output** (green for success, red for failures)
- 📈 **Progress tracking** with status indicators
- 🎯 **Real use cases** with descriptive naming

## Running Demos Locally

All demo scripts are available in the `demo_run/` directory:

```bash
# Clone the repository
git clone https://github.com/ENDEVSOLS/LongProbe.git
cd LongProbe

# Set up demo environment
bash demos/setup_demo_env.sh

# Run any demo
python demo_run/complete_workflow.py
python demo_run/monitor_rag_quality.py
python demo_run/detect_regressions.py
```

## Use Cases Covered

| Demo | Use Case | When to Use |
|------|----------|-------------|
| Complete Workflow | Full workflow demo | Learning LongProbe, understanding all features |
| Monitor RAG Quality | Detailed analysis | Production monitoring, debugging issues |
| Detect Regressions | Safety checks | Before deployment, in CI/CD pipelines |

## Next Steps

Choose a demo to explore:

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } __Complete Workflow__

    ---

    Full workflow with auto-capture and regression detection

    [:octicons-arrow-right-24: View Demo](complete-workflow.md)

-   :material-monitor-dashboard:{ .lg .middle } __Monitor RAG Quality__

    ---

    Detailed monitoring with comprehensive results

    [:octicons-arrow-right-24: View Demo](monitor-quality.md)

-   :material-alert-decagram:{ .lg .middle } __Detect Regressions__

    ---

    Baseline comparison with deployment verdict

    [:octicons-arrow-right-24: View Demo](detect-regressions.md)

</div>
