"""
Check if documents from 7_missing_neo4j_all.csv exist in S3 bucket.
"""
import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS credentials from environment
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET = os.getenv("S3_BUCKET", "s3-lexport-dev-v1")
PREFIX = os.getenv("S3_PREFIX", "neo4j-dev/")

# Documents to check
docs_to_check = [
    "PP-NASIONAL-16-2021",
    "PP-NASIONAL-34-2021",
    "PERMEN-NASIONAL-2-2020",
    "PERMEN-NASIONAL-24-2021",
    "PERMEN-NASIONAL-7-2023",
    "PERMEN-NASIONAL-8-2022",
    "PERPPU-NASIONAL-2-2022",
    "UU-NASIONAL-11-2020",
    "UU-NASIONAL-12-2011",
    "UU-NASIONAL-20-2008",
    "UU-NASIONAL-5-1999",
]

print(f"Connecting to S3 bucket: {BUCKET}")
print(f"Directory: {PREFIX}\n")

# Create S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# List all objects in the neo4j-dev prefix
print("Fetching S3 objects...")
try:
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX)
    
    all_keys = []
    for page in pages:
        if 'Contents' in page:
            all_keys.extend([obj['Key'] for obj in page['Contents']])
    
    print(f"Total objects in {PREFIX}: {len(all_keys)}\n")
    
    # Show first 20 file paths to understand structure
    print("Sample file paths (first 20):")
    for i, key in enumerate(all_keys[:20], 1):
        print(f"  {i}. {key}")
    print()
    
    # Extract doc_ids from S3 keys
    # Try multiple patterns to find doc_ids
    s3_doc_ids = set()
    
    for key in all_keys:
        # Remove prefix
        relative_path = key.replace(PREFIX, '')
        
        # Pattern 1: neo4j-dev/DOC_ID/...
        if '/' in relative_path:
            parts = relative_path.split('/')
            # The doc_id might be in the first part after prefix
            potential_doc_id = parts[0]
            if potential_doc_id and '-' in potential_doc_id:  # doc_ids have dashes
                s3_doc_ids.add(potential_doc_id)
        
        # Pattern 2: filename contains doc_id (e.g., PP-NASIONAL-16-2021.json)
        filename = key.split('/')[-1]
        if filename:
            # Remove extensions
            name_without_ext = filename.replace('.json', '').replace('.csv', '').replace('.txt', '')
            if '-' in name_without_ext and len(name_without_ext.split('-')) >= 4:
                s3_doc_ids.add(name_without_ext)
    
    print(f"Unique doc_ids found in S3: {len(s3_doc_ids)}\n")
    
    # Show first few doc_ids as examples
    print("Sample doc_ids in S3 (first 20):")
    for i, doc_id in enumerate(sorted(s3_doc_ids)[:20], 1):
        print(f"  {i}. {doc_id}")
    print()
    
    # Check each target document
    print("=" * 70)
    print("CHECKING TARGET DOCUMENTS")
    print("=" * 70)
    
    found_count = 0
    missing_count = 0
    
    for doc_id in docs_to_check:
        if doc_id in s3_doc_ids:
            print(f"✅ FOUND:   {doc_id}")
            found_count += 1
        else:
            print(f"❌ MISSING: {doc_id}")
            missing_count += 1
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {found_count} found, {missing_count} missing out of {len(docs_to_check)} documents")
    print("=" * 70)

except Exception as e:
    print(f"Error: {e}")
