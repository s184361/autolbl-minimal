# AutoLbl Restructuring Plan

## Overview
This document outlines the plan to restructure the AutoLbl codebase to follow best practices similar to the Hyperiax project structure.

## Files to Remove/Archive

### ❌ Delete (Obsolete/Temporary)
- `main.py` - Old main script (replaced by run_any3.py)
- `opt_ax_parallel.py` - Experimental optimization script
- `prompt_examples_f1_small.csv` - Temporary data file
- `prompt_examples_f1_small.csv.partial` - Incomplete file
- `mipro_wood_detr_loss_optimized.json` - Experiment result file
- `_annotations.coco.json` - Temporary annotation file
- `uv.lock` - Duplicate lock file (keep if using uv, otherwise delete)

### 📝 Move to `examples/` folder (cleaned up)
- `Florence2_cap.py` - Florence-2 caption and grounding example (CLEANED UP)

### 📦 Move to `experiments/` folder
- `dspy_opt_local.py` - DSPy optimization experiments
- `dspy_prompt_optimizer.py` - Prompt optimization experiments  
- `dspy_qwen_direct.py` - Qwen direct experiments
- `meta_emb_anomaly.py` - Embedding anomaly experiments

### 📂 Folders to Keep (but update .gitignore)
- `data/` - Dataset storage (should be .gitignored)
- `dataset/` - Processed datasets (should be .gitignored)
- `samples/` - Output samples (should be .gitignored)
- `wandb/` - Weights & Biases logs (should be .gitignored)
- `Image_Embeddings/` - Cached embeddings (should be .gitignored)
- `croped_images/` - Processed images (should be .gitignored)
- `optimized_prompts/` - Experiment outputs (should be .gitignored or moved to experiments/)

## Restructuring Steps

### Phase 1: Create New Structure
1. Create `autolbl/` package directory
2. Create subdirectories: `models/`, `data/`, `evaluation/`, `ontology/`, `visualization/`
3. Create `examples/`, `experiments/`, `scripts/`, `tests/`, `docs/` directories
4. Add `__init__.py` files to all package directories

### Phase 2: Move Core Code
1. **Models** (utils/ → autolbl/models/)
   - `Florence_fixed.py` → `florence.py`
   - `grounding_dino_model.py` → `grounding_dino.py`
   - `qwen25_model.py` → `qwen.py`
   - `metaclip_model.py` → `metaclip.py`
   - `metaclip_model_classifier.py` → integrate into `metaclip.py`
   - `composed_detection_model.py` → `composed.py`
   - `grounded_sam.py` → `sam.py`
   - `vlm.py` → keep as base class
   - `detection_base_model.py` → integrate into models `__init__.py`
   - `classification_base_model.py` → integrate into models `__init__.py`

2. **Data Processing** (utils/ → autolbl/data/)
   - `dataset_preparation.py` → `dataset_prep.py`
   - `convert_annotations.py` → `converters.py`
   - `data_processing.py` → `processing.py`

3. **Evaluation** (utils/ → autolbl/evaluation/)
   - `check_labels.py` → `metrics.py`

4. **Ontology** (utils/ → autolbl/ontology/)
   - `embedding_ontology.py` → `embedding.py`

5. **Visualization** (utils/ → autolbl/visualization/)
   - `wandb_utils.py` → `wandb.py`

6. **Core** 
   - Keep `core.py` in `autolbl/core.py`

### Phase 3: Move Scripts
1. **Main Scripts** (root → scripts/)
   - `run_any3.py` → `scripts/run_inference.py`
   - `prepare_datasets.py` → `scripts/prepare_datasets.py`

2. **Experiments** (root → experiments/)
   - Move DSPy and optimization scripts to appropriate subdirectories

### Phase 4: Create Documentation Structure
1. Create `docs/source/` directory
2. Add Sphinx configuration (`conf.py`)
3. Create API documentation structure
4. Add example notebooks to `docs/source/notebooks/`

### Phase 5: Create Tests
1. Create `tests/` directory with:
   - `test_models.py` - Test vision models
   - `test_data.py` - Test data processing
   - `test_evaluation.py` - Test metrics
   - `fixtures.py` - Shared test fixtures

### Phase 6: Update Imports
1. Update all imports from `utils.*` to `autolbl.*`
2. Update import paths in all scripts
3. Update config.json references if needed

### Phase 7: Update Configuration Files
1. Update `pyproject.toml` with new structure
2. Add `setup.py` for development install
3. Update `.gitignore` to exclude data folders
4. Create `CONTRIBUTION.md`
5. Add `LICENSE` file

### Phase 8: Update Documentation
1. Update README.md with new structure
2. Update DATASET_PREPARATION.md with new script paths
3. Add API documentation
4. Create usage examples in examples/

## Benefits of New Structure

### ✅ Clarity
- Clear separation of concerns (models, data, evaluation)
- Experimental code isolated from production code
- Examples separate from core library

### ✅ Maintainability
- Easier to find and modify specific components
- Better organization for collaborative development
- Clear testing structure

### ✅ Professional
- Follows Python packaging best practices
- Similar to well-established projects (Hyperiax, scikit-learn)
- Easier for others to contribute

### ✅ Documentation
- Sphinx-ready structure for auto-generated docs
- Examples in dedicated folder
- Clear separation of tutorials and API docs

## Implementation Checklist

- [ ] Phase 1: Create directories
- [ ] Phase 2: Move and rename files
- [ ] Phase 3: Move scripts
- [ ] Phase 4: Setup docs structure
- [ ] Phase 5: Create test structure
- [ ] Phase 6: Update all imports
- [ ] Phase 7: Update config files
- [ ] Phase 8: Update documentation
- [ ] Delete obsolete files
- [ ] Test all functionality
- [ ] Update .gitignore
- [ ] Commit and push changes

## Migration Command Summary

```bash
# Create new structure
mkdir -p autolbl/{models,data,evaluation,ontology,visualization}
mkdir -p examples experiments/prompt_optimization experiments/embedding
mkdir -p scripts tests docs/source

# Add __init__.py files
touch autolbl/__init__.py
touch autolbl/models/__init__.py
touch autolbl/data/__init__.py
touch autolbl/evaluation/__init__.py
touch autolbl/ontology/__init__.py
touch autolbl/visualization/__init__.py
touch tests/__init__.py

# Move files (examples - adapt for Windows)
# Models
move utils\Florence_fixed.py autolbl\models\florence.py
move utils\grounding_dino_model.py autolbl\models\grounding_dino.py
move utils\qwen25_model.py autolbl\models\qwen.py
# ... etc

# Move scripts
move run_any3.py scripts\run_inference.py
move prepare_datasets.py scripts\prepare_datasets.py

# Move experiments
move dspy_opt_local.py experiments\prompt_optimization\
move dspy_prompt_optimizer.py experiments\prompt_optimization\
move meta_emb_anomaly.py experiments\embedding\

# Delete obsolete
del main.py
del Florence2_cap.py
del opt_ax_parallel.py
```

## Notes
- Keep `config.json` in root for backward compatibility
- Data folders stay in root but add to .gitignore
- Old utils/ folder can be deleted after migration
- Create git branches for safe migration
