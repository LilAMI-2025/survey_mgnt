#!/usr/bin/env python3
"""
Test script to verify the fixed pipeline works locally
"""

import os
import sys

def test_imports():
    """Test that all imports work"""
    print("🧪 Testing imports...")
    try:
        from colab_gsheets_deduplicator import (
            SnowflakeConnection, 
            SurveyDeduplicator,
            quick_full_pipeline_with_sheets_export
        )
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_snowflake_connection():
    """Test Snowflake connection with fixed role"""
    print("\n🔍 Testing Snowflake connection...")
    try:
        from colab_gsheets_deduplicator import SnowflakeConnection
        
        # Set test credentials
        os.environ['SNOWFLAKE_USER'] = 'ami_tableau'
        os.environ['SNOWFLAKE_PASSWORD'] = 'fok7domp6CEED!auw'
        os.environ['SNOWFLAKE_ACCOUNT'] = 'qu54429.eu-central-1'
        
        conn = SnowflakeConnection()
        engine = conn.connect()
        
        print("✅ Snowflake connection successful!")
        return True
    except Exception as e:
        print(f"❌ Snowflake connection failed: {e}")
        return False

def test_deduplication_logic():
    """Test the fixed deduplication sorting logic"""
    print("\n🔧 Testing deduplication logic...")
    try:
        import pandas as pd
        from colab_gsheets_deduplicator import SurveyDeduplicator
        
        # Create test data
        test_data = {
            'question_text': ['What is your business type?'] * 3,
            'choice_text': ['Small', 'Medium Business', 'Large Corporation'],
            'uid': ['uid1', 'uid2', 'uid3'],
            'survey_stage': ['Test'] * 3,
            'date_modified': ['2024-01-01'] * 3
        }
        
        df = pd.DataFrame(test_data)
        dedup = SurveyDeduplicator()
        
        result_df, report = dedup.apply_enhanced_deduplication_solutions(df)
        
        # Check that choice_text is still text, not numbers
        choice_types = result_df['choice_text'].apply(type).unique()
        if all(isinstance(x, str) for x in result_df['choice_text']):
            print("✅ Deduplication logic working - choice_text preserved as strings!")
            return True
        else:
            print(f"❌ Issue: choice_text types are {choice_types}")
            return False
            
    except Exception as e:
        print(f"❌ Deduplication test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Fixed Survey Deduplication Pipeline")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_snowflake_connection,
        test_deduplication_logic
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! Your local pipeline is ready.")
        print("\n📋 Next steps:")
        print("1. Commit changes: git add . && git commit -m 'Applied all fixes'")
        print("2. Push to GitHub: git push origin main")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 