"""Interactive test for autolbl package functionality."""

import sys
from pathlib import Path

def test_package_structure():
    """Test that the package structure is correct."""
    print("=" * 60)
    print("Testing AutoLbl Package Structure")
    print("=" * 60)
    
    try:
        import autolbl
        print(f"✅ Package version: {autolbl.__version__}")
        print(f"✅ Package location: {autolbl.__file__}")
        
        # Test submodules exist
        print("\n📦 Testing submodules...")
        assert hasattr(autolbl, 'models'), "Missing models module"
        print("  ✅ autolbl.models")
        
        assert hasattr(autolbl, 'datasets'), "Missing datasets module"
        print("  ✅ autolbl.datasets")
        
        assert hasattr(autolbl, 'evaluation'), "Missing evaluation module"
        print("  ✅ autolbl.evaluation")
        
        assert hasattr(autolbl, 'ontology'), "Missing ontology module"
        print("  ✅ autolbl.ontology")
        
        assert hasattr(autolbl, 'visualization'), "Missing visualization module"
        print("  ✅ autolbl.visualization")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_imports():
    """Test that model classes can be imported."""
    print("\n" + "=" * 60)
    print("Testing Model Imports")
    print("=" * 60)
    
    models_to_test = [
        ("Florence2", "autolbl.models.florence"),
        ("GroundingDINO", "autolbl.models.grounding_dino"),
        ("Qwen25VL", "autolbl.models.qwen"),
        ("MetaCLIP", "autolbl.models.metaclip"),
        ("ComposedDetectionModel2", "autolbl.models.composed"),
    ]
    
    success = True
    for class_name, module_path in models_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✅ {class_name} from {module_path}")
        except ImportError as e:
            print(f"  ⚠️  {class_name} - Import error (may need optional dependencies): {e}")
        except Exception as e:
            print(f"  ❌ {class_name} - Error: {e}")
            success = False
    
    return success

def test_dataset_utilities():
    """Test dataset utility functions."""
    print("\n" + "=" * 60)
    print("Testing Dataset Utilities")
    print("=" * 60)
    
    try:
        from autolbl.datasets.dataset_prep import get_base_dir
        
        # Test base directory detection
        base_dir = get_base_dir(__file__)
        print(f"  ✅ Base directory detected: {base_dir}")
        
        # Check if it found the project root correctly
        pyproject = base_dir / "pyproject.toml"
        if pyproject.exists():
            print(f"  ✅ Found pyproject.toml at project root")
        else:
            print(f"  ⚠️  pyproject.toml not found - base_dir may be incorrect")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_metrics():
    """Test evaluation metrics."""
    print("\n" + "=" * 60)
    print("Testing Evaluation Metrics")
    print("=" * 60)
    
    try:
        from autolbl.evaluation.metrics import evaluate_detections
        print("  ✅ evaluate_detections imported")
        
        # You could add a simple test with dummy data here
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_availability():
    """Test that CLI commands are available."""
    print("\n" + "=" * 60)
    print("Testing CLI Availability")
    print("=" * 60)
    
    try:
        from autolbl.cli.prepare import main as prepare_main
        from autolbl.cli.infer import main as infer_main
        
        print("  ✅ autolbl-prepare (prepare_main)")
        print("  ✅ autolbl-infer (infer_main)")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n🚀 AutoLbl Package Testing Suite\n")
    
    results = {
        "Package Structure": test_package_structure(),
        "Model Imports": test_model_imports(),
        "Dataset Utilities": test_dataset_utilities(),
        "Evaluation Metrics": test_evaluation_metrics(),
        "CLI Availability": test_cli_availability(),
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! Package restructuring successful!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
