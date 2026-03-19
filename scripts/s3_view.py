import boto3
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

s3 = boto3.client('s3', region_name='ap-southeast-3')
bucket = 's3-lexport-dev-v1'
prefix = 'neo4j-dev/'

all_rows = []

paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

for page in pages:
    for obj in page.get('Contents', []):
        key = obj['Key']
        if key.endswith('/') or key == prefix:
            continue

        filename = key.replace(prefix, '')
        size = obj['Size']
        last_modified = obj['LastModified']

        # Make datetime timezone-unaware for Excel
        if hasattr(last_modified, "tzinfo") and last_modified.tzinfo is not None:
            last_modified = last_modified.replace(tzinfo=None)

        all_rows.append({
            'key': key,
            'filename': filename,
            'size_bytes': size,
            'last_modified': last_modified,
        })

print(f"Found {len(all_rows)} files in {bucket}/{prefix}")

output_path = r'E:\Files\Projects\UU_new\data\s3_listing.xlsx'
df = pd.DataFrame(all_rows)

# Ensure all datetimes are timezone-unaware
for col in df.select_dtypes(include=["datetimetz"]).columns:
    df[col] = df[col].dt.tz_localize(None)

df.to_excel(output_path, index=False)

print(f"✓ Saved listing to {output_path}")
