import re

def extract_dataset_name(s3_url):
    if not s3_url:
        return "unknown-dataset"

    # Extract filename from URL
    filename = s3_url.split('/')[-1]

    # Remove extension
    base_name = filename.rsplit('.', 1)[0]

    # Remove banned words (case-insensitive)
    words_to_remove = ['homogeneous', 'heterogeneous', 'trees', 'tree']
    for word in words_to_remove:
        base_name = re.sub(f'(?i){word}', '', base_name)

    # Clean up underscores
    base_name = re.sub(r'_+', '_', base_name)
    base_name = base_name.strip('_')

    return base_name if base_name else "unknown-dataset"