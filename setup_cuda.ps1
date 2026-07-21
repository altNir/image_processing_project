param(
    [string]$Python = "python",
    [ValidateSet("cu126", "cu130", "cu132")]
    [string]$CudaWheel = "cu130"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Write-Host "Checking the selected Python interpreter..."
& $Python -c "import sys; print(sys.executable); print(sys.version)"

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "NVIDIA driver detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} else {
    throw "nvidia-smi was not found. Install/update the NVIDIA driver before CUDA PyTorch."
}

$indexUrl = "https://download.pytorch.org/whl/$CudaWheel"
Write-Host "Installing the PyTorch $CudaWheel wheel from $indexUrl ..."
& $Python -m pip install --upgrade --force-reinstall torch torchvision --index-url $indexUrl

$cudaCheck = @'
import sys
import torch

print('Python:', sys.executable)
print('PyTorch:', torch.__version__)
print('Wheel CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA is still unavailable. Confirm this is the same Python used to run the project.')
print('GPU:', torch.cuda.get_device_name(0))
properties = torch.cuda.get_device_properties(0)
print('GPU memory (GiB):', round(properties.total_memory / 1024**3, 2))
x = torch.randn((1024, 1024), device='cuda', dtype=torch.float16)
print('FP16 CUDA smoke test:', float((x @ x.T).mean()))
'@

& $Python -c $cudaCheck
if ($LASTEXITCODE -ne 0) {
    throw "CUDA verification failed with exit code $LASTEXITCODE."
}
Write-Host "CUDA PyTorch is ready. Run the project with --device cuda."
