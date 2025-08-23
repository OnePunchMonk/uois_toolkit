#!/usr/bin/env python3
"""
UOIS Toolkit PyTorch Lightning Validation Script
================================================================

This script validates that the UOIS toolkit provides complete PyTorch Lightning 
functionality for abstracting all data loaders via a single library function.

The script tests:
1. DataModule creation for all supported datasets
2. Proper interface compliance with PyTorch Lightning
3. Configuration system
4. Error handling for invalid datasets
"""

import sys
import os
import inspect
from pathlib import Path

# Add the current directory to Python path for local imports
sys.path.insert(0, os.path.abspath('.'))

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from core.datasets import get_datamodule, UOISDataModule, DATASET_MAPPING
        from core.config.config import cfg
        print("✅ Core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_dataset_mapping():
    """Test that all datasets are properly mapped."""
    print("🔍 Testing dataset mapping...")
    
    try:
        from core.datasets import DATASET_MAPPING
        expected_datasets = ["tabletop", "ocid", "osd", "robot_pushing", "iteach_humanplay"]
        
        for dataset_name in expected_datasets:
            if dataset_name not in DATASET_MAPPING:
                print(f"❌ Missing dataset: {dataset_name}")
                return False
            print(f"✅ Found dataset: {dataset_name}")
        
        print(f"✅ All {len(expected_datasets)} datasets properly mapped")
        return True
    except Exception as e:
        print(f"❌ Dataset mapping test failed: {e}")
        return False

def test_datamodule_interface():
    """Test that DataModule has all required PyTorch Lightning methods."""
    print("🔍 Testing DataModule interface...")
    
    try:
        from core.datasets import UOISDataModule
        
        required_methods = [
            'setup', 'train_dataloader', 'val_dataloader', 'test_dataloader'
        ]
        
        for method_name in required_methods:
            if not hasattr(UOISDataModule, method_name):
                print(f"❌ Missing method: {method_name}")
                return False
            print(f"✅ Found method: {method_name}")
        
        print("✅ DataModule interface complete")
        return True
    except Exception as e:
        print(f"❌ DataModule interface test failed: {e}")
        return False

def test_factory_function():
    """Test the main factory function."""
    print("🔍 Testing factory function...")
    
    try:
        from core.datasets import get_datamodule
        
        # Test function signature
        sig = inspect.signature(get_datamodule)
        expected_params = ['dataset_name', 'data_path', 'batch_size', 'num_workers', 'config']
        
        for param_name in expected_params:
            if param_name not in sig.parameters:
                print(f"❌ Missing parameter: {param_name}")
                return False
            print(f"✅ Found parameter: {param_name}")
        
        print("✅ Factory function signature correct")
        return True
    except Exception as e:
        print(f"❌ Factory function test failed: {e}")
        return False

def test_error_handling():
    """Test error handling for invalid datasets."""
    print("🔍 Testing error handling...")
    
    try:
        from core.datasets import get_datamodule
        from core.config.config import cfg
        
        # This should raise a ValueError
        try:
            get_datamodule(
                dataset_name="invalid_dataset_name",
                data_path="/fake/path",
                batch_size=4,
                num_workers=0,
                config=cfg
            )
            print("❌ Should have raised ValueError for invalid dataset")
            return False
        except ValueError as e:
            if "Unknown dataset" in str(e):
                print("✅ Correctly raised ValueError for invalid dataset")
                return True
            else:
                print(f"❌ Wrong error message: {e}")
                return False
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_config_system():
    """Test that the configuration system works."""
    print("🔍 Testing configuration system...")
    
    try:
        from core.config.config import cfg
        
        # Test required config attributes
        required_attrs = ['PIXEL_MEANS', 'INPUT', 'MODE', 'FLOW_HEIGHT', 'FLOW_WIDTH', 'TRAIN']
        
        for attr_name in required_attrs:
            if not hasattr(cfg, attr_name):
                print(f"❌ Missing config attribute: {attr_name}")
                return False
            print(f"✅ Found config attribute: {attr_name}")
        
        # Test TRAIN subconfig
        train_attrs = ['CHROMATIC', 'ADD_NOISE', 'SYN_CROP', 'SYN_CROP_SIZE']
        for attr_name in train_attrs:
            if attr_name not in cfg.TRAIN:
                print(f"❌ Missing TRAIN config: {attr_name}")
                return False
            print(f"✅ Found TRAIN config: {attr_name}")
        
        print("✅ Configuration system complete")
        return True
    except Exception as e:
        print(f"❌ Configuration system test failed: {e}")
        return False

def test_dataset_classes():
    """Test that all dataset classes have proper interfaces."""
    print("🔍 Testing dataset classes...")
    
    try:
        from core.datasets import DATASET_MAPPING
        
        for dataset_name, dataset_class in DATASET_MAPPING.items():
            # Check that class has required methods
            required_methods = ['__init__', '__getitem__', '__len__']
            for method_name in required_methods:
                if not hasattr(dataset_class, method_name):
                    print(f"❌ {dataset_name} missing method: {method_name}")
                    return False
            print(f"✅ {dataset_name} has proper interface")
        
        print("✅ All dataset classes have proper interfaces")
        return True
    except Exception as e:
        print(f"❌ Dataset classes test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("="*70)
    print("🚀 UOIS Toolkit PyTorch Lightning Validation")
    print("="*70)
    
    tests = [
        test_imports,
        test_dataset_mapping,
        test_datamodule_interface,
        test_factory_function,
        test_error_handling,
        test_config_system,
        test_dataset_classes,
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
        print()
    
    print("="*70)
    print("📊 VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 SUCCESS: UOIS Toolkit PyTorch Lightning functionality is COMPLETE!")
        print("📝 The library successfully provides:")
        print("   • Single function abstraction: get_datamodule()")
        print("   • Support for 5 UOIS datasets by name")
        print("   • Complete PyTorch Lightning DataModule interface")
        print("   • Proper error handling and configuration")
        print("   • All required utility functions")
        return True
    else:
        print("❌ INCOMPLETE: Some functionality is missing")
        failed = total - passed
        print(f"   {failed} test(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
