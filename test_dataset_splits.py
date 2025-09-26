"""
Test script for the dataset splitting functionality
Tests the splitting logic with real datasets or provides setup instructions
"""

import os
import sys
import pandas as pd

# Import our splitting functions
sys.path.append('.')
try:
    from create_dataset_splits import main, DataProtection
except ImportError:
    print("ERROR: Could not import create_dataset_splits.py")
    print("Make sure both files are in the same directory")
    sys.exit(1)

def test_dataset_splits():
    """Test the dataset splitting functionality"""
    
    print("Testing Dataset Splitting Pipeline")
    print("=" * 50)
    
    # Check if we have a dataset available
    dataset_paths = [
        'covid-chestxray-dataset',
        'data/covid-chestxray-dataset', 
        'data/raw/covid-chestxray-dataset',
        'data/raw/NIH_ChestXray'
    ]
    
    dataset_found = False
    for path in dataset_paths:
        if os.path.exists(os.path.join(path, 'metadata.csv')) or os.path.exists(os.path.join(path, 'Data_Entry_2017_v2020.csv')):
            dataset_found = True
            break
    
    if not dataset_found:
        print("No dataset found. To test with real data:")
        print("")
        print("Download COVID-19 Chest X-ray Dataset (Recommended):")
        print("   git clone https://github.com/ieee8023/covid-chestxray-dataset.git")
        print("")
        print("Or download NIH Chest X-ray Dataset:")
        print("   https://nihcc.app.box.com/v/ChestXray-NIHCC")
        print("")
        print("Place dataset in current directory or 'data/' folder")
        print("")
        print("For now, testing the code structure...")
        
        # Test that the functions exist and are callable
        print("")
        print("Testing function imports...")
        assert callable(main), "main() function should be callable"
        print("main() function imported successfully")
        
        try:
            # Try to run main (will show download instructions)
            result = main()
            if result is None:
                print("main() correctly returns None when no dataset found")
        except Exception as e:
            print(f"Error in main(): {e}")
            return False
        
        print("")
        print("Code structure test passed!")
        print("Download a dataset to test full functionality")
        return True
    
    else:
        print("Dataset found! Running full test...")
        
        # Run the main splitting function
        try:
            loader = main()
            
            if loader is None:
                print("ERROR: main() returned None despite dataset being found")
                return False
            
            print("")
            print("Testing data access...")
            
            # Test train and test data access
            train_data = loader.get_train_data()
            test_data = loader.get_test_data()
            
            print(f"Train data: {len(train_data)} samples")
            print(f"Test data: {len(test_data)} samples")
            
            # Test validation data protection
            print("")
            print("Testing validation data protection...")
            try:
                val_data = loader.get_validation_data()
                print("ERROR: Validation data access should be denied without confirmation!")
                return False
            except ValueError as e:
                print("Validation data protection works correctly")
                print(f"   Error message: {str(e).split()[0:6]} ...")
            
            # Test authorized validation access
            print("")
            print("Testing authorized validation access...")
            try:
                val_data = loader.get_validation_data(confirm_final_evaluation=True)
                print(f"Authorized validation data access: {len(val_data)} samples")
            except Exception as e:
                print(f"Error in validation data access: {e}")
                return False
            
            # Verify split ratios
            total = len(train_data) + len(test_data) + len(val_data)
            train_pct = len(train_data) / total * 100
            test_pct = len(test_data) / total * 100
            val_pct = len(val_data) / total * 100
            
            print(f"")
            print(f"Split verification:")
            print(f"   Training: {train_pct:.1f}% (target: ~70%)")
            print(f"   Test: {test_pct:.1f}% (target: ~15%)")
            print(f"   Validation: {val_pct:.1f}% (target: ~15%)")
            
            # Check if ratios are approximately correct (within 5%)
            if 65 <= train_pct <= 75 and 10 <= test_pct <= 20 and 10 <= val_pct <= 20:
                print("Split ratios are within acceptable range")
            else:
                print("WARNING: Split ratios are outside target range")
            
            print("")
            print("Full functionality test passed!")
            return True
            
        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_dataset_splits()
    if success:
        print("")
        print("All tests passed!")
    else:
        print("")
        print("Some tests failed!")
        sys.exit(1)
