"""Quick test to verify all imports work after restructuring."""

def test_imports():
    """Test that all key modules can be imported."""
    
    print("Testing imports...")
    
    # Test core package
    print("  ✓ Testing autolbl...")
    import autolbl
    print(f"    Version: {autolbl.__version__}")
    
    # Test models
    print("  ✓ Testing autolbl.models...")
    from autolbl.models.florence import Florence2
    from autolbl.models.grounding_dino import GroundingDINO
    from autolbl.models.qwen import Qwen25VL
    from autolbl.models.metaclip import MetaCLIP
    from autolbl.models.composed import ComposedDetectionModel2
    
    # Test data utilities
    print("  ✓ Testing autolbl.datasets...")
    from autolbl.datasets.dataset_prep import get_base_dir, update_config_section
    from autolbl.datasets.converters import convert_bbox_annotation
    
    # Test evaluation
    print("  ✓ Testing autolbl.evaluation...")
    from autolbl.evaluation.metrics import evaluate_detections
    
    # Test ontology
    print("  ✓ Testing autolbl.ontology...")
    from autolbl.ontology.embedding import EmbeddingOntologyImage
    
    # Test visualization
    print("  ✓ Testing autolbl.visualization...")
    from autolbl.visualization.wandb import compare_plot
    
    # Test CLI
    print("  ✓ Testing autolbl.cli...")
    from autolbl.cli.prepare import main as prepare_main
    from autolbl.cli.infer import main as infer_main
    
    print("\n✅ All imports successful!")
    return True

if __name__ == "__main__":
    try:
        test_imports()
        print("\n🎉 Package restructuring verified successfully!")
    except Exception as e:
        print(f"\n❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
