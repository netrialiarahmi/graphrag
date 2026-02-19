"""
Download JSON files for documents that are in S3 but missing from Neo4j.
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

# Documents to download
docs_to_download = [
    "PP-NASIONAL-16-2021",
    "PP-NASIONAL-34-2021",
    "PERMEN-NASIONAL-8-2022",
    "UU-NASIONAL-11-2020",
]

# Create output directory
OUTPUT_DIR = "benchmark/downloaded_s3_docs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Downloading JSON files from S3 bucket: {BUCKET}")
print(f"Output directory: {OUTPUT_DIR}\n")

# Create S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Download each JSON file
success_count = 0
failed_count = 0

for doc_id in docs_to_download:
    s3_key = f"{PREFIX}{doc_id}.json"
    local_path = os.path.join(OUTPUT_DIR, f"{doc_id}.json")
    
    try:
        print(f"Downloading: {s3_key}")
        s3.download_file(BUCKET, s3_key, local_path)
        
        # Get file size
        file_size = os.path.getsize(local_path)
        print(f"  ✅ Saved to: {local_path} ({file_size:,} bytes)")
        success_count += 1
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        failed_count += 1
    
    print()

print("=" * 70)
print(f"SUMMARY: {success_count} downloaded, {failed_count} failed")
print("=" * 70)
