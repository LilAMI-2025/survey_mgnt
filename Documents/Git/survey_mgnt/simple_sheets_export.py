"""
Simple, bulletproof Google Sheets export for survey deduplication results
This bypasses the complex formatting that's causing API errors.
"""

import gspread
from google.auth import default
import pandas as pd
import re
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def export_results_to_sheets_simple(step1a_df: pd.DataFrame, step1b_df: pd.DataFrame, 
                                   step1a_report: Dict, step1b_report: Dict, 
                                   sheets_url: str) -> Dict[str, str]:
    """
    Simple, reliable export to Google Sheets that avoids API formatting issues
    """
    try:
        # Extract sheet ID from URL
        match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', sheets_url)
        if not match:
            raise ValueError("Invalid Google Sheets URL")
        sheet_id = match.group(1)
        
        # Authenticate
        creds, _ = default()
        gc = gspread.authorize(creds)
        
        # Open spreadsheet
        spreadsheet = gc.open_by_key(sheet_id)
        
        # === Export Step 1A Results ===
        print("📤 Exporting Step 1A results...")
        try:
            ws1a = spreadsheet.worksheet('Step1A_Results_Simple')
            ws1a.clear()
        except gspread.WorksheetNotFound:
            ws1a = spreadsheet.add_worksheet(title='Step1A_Results_Simple', rows=2000, cols=10)
        
        # Prepare Step 1A data
        step1a_clean = step1a_df.fillna('').astype(str)
        step1a_data = [
            [f"Step 1A: Question Bank Deduplication - {step1a_report['reduction_percentage']:.2f}% Reduction"],
            [],  # Empty row
            step1a_clean.columns.tolist()  # Headers
        ] + step1a_clean.values.tolist()  # Data
        
        # Export Step 1A (simple update)
        ws1a.update('A1', step1a_data)
        
        # === Export Step 1B Results ===
        print("📤 Exporting Step 1B results...")
        try:
            ws1b = spreadsheet.worksheet('Step1B_Results_Simple')
            ws1b.clear()
        except gspread.WorksheetNotFound:
            ws1b = spreadsheet.add_worksheet(title='Step1B_Results_Simple', rows=2000, cols=10)
        
        # Prepare Step 1B data
        step1b_clean = step1b_df.fillna('').astype(str)
        step1b_data = [
            [f"Step 1B: Comprehensive Deduplication - {step1b_report['reduction_percentage']:.2f}% Reduction"],
            [],  # Empty row
            step1b_clean.columns.tolist()  # Headers
        ] + step1b_clean.values.tolist()  # Data
        
        # Export Step 1B (simple update)
        ws1b.update('A1', step1b_data)
        
        # === Export Summary ===
        print("📤 Exporting summary...")
        try:
            ws_summary = spreadsheet.worksheet('Summary_Simple')
            ws_summary.clear()
        except gspread.WorksheetNotFound:
            ws_summary = spreadsheet.add_worksheet(title='Summary_Simple', rows=100, cols=5)
        
        summary_data = [
            ['Survey Question Bank Deduplication Summary'],
            [],
            ['Step 1A Results:'],
            ['Original Records:', str(step1a_report['original_count'])],
            ['Final Records:', str(step1a_report['final_count'])],
            ['Reduction Percentage:', f"{step1a_report['reduction_percentage']:.2f}%"],
            [],
            ['Step 1B Results:'],
            ['Original Records:', str(step1b_report['original_count'])],
            ['Final Records:', str(step1b_report['final_count'])],
            ['Reduction Percentage:', f"{step1b_report['reduction_percentage']:.2f}%"],
            [],
            ['Performance:'],
            ['Step 1A Status:', 'Exceptional (99.88% target)' if step1a_report['reduction_percentage'] > 99 else 'Good'],
            ['Step 1B Status:', 'Exceptional (99.66% target)' if step1b_report['reduction_percentage'] > 99 else 'Good']
        ]
        
        ws_summary.update('A1', summary_data)
        
        print("✅ Successfully exported all results to Google Sheets!")
        
        return {
            'spreadsheet_url': f"https://docs.google.com/spreadsheets/d/{sheet_id}",
            'step1a_count': len(step1a_df),
            'step1b_count': len(step1b_df),
            'export_timestamp': pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Simple export failed: {e}")
        raise

def quick_pipeline_with_simple_export(sheets_url: str) -> Dict[str, Any]:
    """
    Run the complete pipeline with simple, reliable Google Sheets export
    """
    from colab_gsheets_deduplicator import ColabSurveyDeduplicationPipeline
    
    print("🚀 Running pipeline with simple export...")
    
    # Initialize pipeline
    pipeline = ColabSurveyDeduplicationPipeline(sheets_url=sheets_url)
    
    # Run Step 1A
    print("📊 Running Step 1A deduplication...")
    step1a = pipeline.run_step1a_deduplication()
    
    # Run Step 1B  
    print("📊 Running Step 1B deduplication...")
    step1b = pipeline.run_step1b_deduplication()
    
    # Simple export
    print("📤 Exporting with simple method...")
    export_result = export_results_to_sheets_simple(
        step1a['dataframe'],
        step1b['dataframe'], 
        step1a['report'],
        step1b['report'],
        sheets_url
    )
    
    # Get summary
    summary = pipeline.get_summary_report()
    
    return {
        'step1a_reduction': step1a['report']['reduction_percentage'],
        'step1b_reduction': step1b['report']['reduction_percentage'],
        'step1a_records': len(step1a['dataframe']),
        'step1b_records': len(step1b['dataframe']),
        'export_url': export_result['spreadsheet_url'],
        'summary': summary
    }

if __name__ == "__main__":
    print("🔧 Simple Google Sheets Export Module")
    print("Use: quick_pipeline_with_simple_export(sheets_url)") 