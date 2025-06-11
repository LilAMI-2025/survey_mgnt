"""
Google Colab Setup Script for Survey Deduplication with Sheets Export
Run this first in your Colab notebook to set up everything needed.

Usage:
    exec(open('colab_setup.py').read())
    
Or copy and paste this entire script into a Colab cell.
"""

import subprocess
import sys
import os

def install_packages():
    """Install required packages for the deduplication pipeline"""
    packages = [
        'snowflake-connector-python',
        'sqlalchemy',
        'gspread',
        'google-auth',
        'google-auth-oauthlib', 
        'google-auth-httplib2',
        'openpyxl'
    ]
    
    print("📦 Installing required packages...")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    print("✅ All packages installed successfully!")

def setup_authentication():
    """Set up Google authentication for Sheets access"""
    try:
        from google.colab import auth
        auth.authenticate_user()
        print("✅ Google authentication completed!")
        return True
    except ImportError:
        print("⚠️ Not in Google Colab environment. Manual authentication required.")
        return False

def download_deduplication_module():
    """Download or create the deduplication module"""
    module_code = '''
# This would contain the complete colab_gsheets_deduplicator.py content
# For the demo, we'll create a minimal version
print("🚀 Deduplication module loaded!")
'''
    
    with open('colab_deduplication.py', 'w') as f:
        f.write(module_code)
    
    print("📝 Deduplication module created!")

def configure_credentials():
    """Help configure Snowflake credentials"""
    from getpass import getpass
    
    print("🔐 Configuring Snowflake credentials...")
    
    # Default configuration based on the working system
    default_config = {
        'user': 'ami_tableau',
        'account': 'qu54429.eu-central-1', 
        'warehouse': 'COMPUTE_WH',
        'database': 'AMI_DBT',
        'schema': 'DBT_SURVEY_MONKEY',
        'role': 'ACCOUNTADMIN'
    }
    
    print("📋 Using default Snowflake configuration:")
    for key, value in default_config.items():
        print(f"   {key}: {value}")
    
    password = getpass("Enter Snowflake password: ")
    default_config['password'] = password
    
    print("✅ Snowflake credentials configured!")
    return default_config

def setup_google_sheets():
    """Help configure Google Sheets URL"""
    print("📊 Google Sheets Configuration")
    print("🔗 Your Google Sheets URL:")
    print("   https://docs.google.com/spreadsheets/d/19KIcL55rTOeaGFNp0SJvjWY4ps_J8xkn8B_EVJ5c0_k/edit?gid=0#gid=0")
    
    sheets_url = input("Enter your Google Sheets URL (or press Enter to use default): ").strip()
    
    if not sheets_url:
        sheets_url = "https://docs.google.com/spreadsheets/d/19KIcL55rTOeaGFNp0SJvjWY4ps_J8xkn8B_EVJ5c0_k/edit?gid=0#gid=0"
    
    print(f"📝 Using Google Sheets: {sheets_url}")
    return sheets_url

def main():
    """Main setup function"""
    print("🎯 Survey Deduplication Pipeline Setup for Google Colab")
    print("=" * 60)
    
    # Step 1: Install packages
    install_packages()
    print()
    
    # Step 2: Set up authentication  
    auth_success = setup_authentication()
    print()
    
    # Step 3: Download/create deduplication module
    download_deduplication_module()
    print()
    
    # Step 4: Configure credentials
    snowflake_creds = configure_credentials()
    print()
    
    # Step 5: Configure Google Sheets
    sheets_url = setup_google_sheets()
    print()
    
    print("🎉 Setup completed successfully!")
    print("📋 Next steps:")
    print("   1. Import the deduplication pipeline")
    print("   2. Run Step 1A deduplication (99.88% reduction target)")
    print("   3. Run Step 1B deduplication (99.66% reduction target)")
    print("   4. Export results to Google Sheets")
    print()
    print("💡 Quick start code:")
    print("   from colab_gsheets_deduplicator import quick_full_pipeline_with_sheets_export")
    print(f"   results = quick_full_pipeline_with_sheets_export('{sheets_url}', snowflake_credentials)")
    
    return {
        'snowflake_credentials': snowflake_creds,
        'sheets_url': sheets_url,
        'auth_success': auth_success
    }

if __name__ == "__main__":
    config = main()
    
    # Make configuration available globally
    globals()['SNOWFLAKE_CREDENTIALS'] = config['snowflake_credentials']
    globals()['GOOGLE_SHEETS_URL'] = config['sheets_url']
    
    print("\n✅ Configuration saved to global variables:")
    print("   SNOWFLAKE_CREDENTIALS")
    print("   GOOGLE_SHEETS_URL") 