import kagglehub

# Download latest version
path = kagglehub.dataset_download("minhtranv/msu-mfsd-processed-into-frames")

print("Path to dataset files:", path)