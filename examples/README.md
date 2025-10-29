# AutoLbl Examples

This folder contains example scripts demonstrating various AutoLbl capabilities.

## Available Examples

### `florence_captioning.py`
Demonstrates using Florence-2 for:
- Generating detailed image captions
- Caption-to-phrase grounding (detecting objects mentioned in captions)
- Logging results to Weights & Biases

**Usage:**
```bash
python examples/florence_captioning.py
```

**Note:** Edit the script to point to your image folder.

## Adding New Examples

When creating new examples:
1. Add clear docstrings explaining the purpose
2. Include usage instructions
3. Use relative imports: `from autolbl.models import ...`
4. Add example to this README
