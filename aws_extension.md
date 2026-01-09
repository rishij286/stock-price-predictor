# AWS Cloud Extension

This project includes a small cloud-based extension using AWS.

## What it does
- Stores input data in **Amazon S3**
- Uses an **Amazon EC2** free-tier instance to run preprocessing / scripts
- Uploads generated outputs back to **S3**

## Example S3 structure
- `s3://<bucket>/data/` (inputs)
- `s3://<bucket>/outputs/` (results)

## Notes
Built as a hands-on cloud learning extension while preparing for AWS Cloud Practitioner.