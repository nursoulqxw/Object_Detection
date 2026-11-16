import kagglehub

path = kagglehub.dataset_download(
    "ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes"
)
print("Path to dataset files:", path)