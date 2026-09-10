[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$DeploymentId,

    [Parameter(Mandatory = $true)]
    [string]$EvidencePath,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$BoardHost,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$BoardUser,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^/[A-Za-z0-9._/-]+$')]
    [string]$RemoteRoot,

    [string]$ArtifactDir = 'build/vivado/npu_matrix_8x8/artifacts',

    [string]$ModelDir = 'examples/resnet18/model',

    [switch]$AllowArtifactCommitMismatch,

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-CommandAvailable {
    param([Parameter(Mandatory = $true)][string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command is unavailable: $Name"
    }
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$resolvedArtifacts = (Resolve-Path (Join-Path $repositoryRoot $ArtifactDir)).Path
$resolvedModel = (Resolve-Path (Join-Path $repositoryRoot $ModelDir)).Path
$resolvedEvidence = [IO.Path]::GetFullPath(
    (Join-Path $repositoryRoot $EvidencePath)
)

$requiredArtifacts = @(
    'npu_matrix.bit',
    'npu_matrix.hwh',
    'npu_matrix.manifest.json'
)
$requiredModel = @(
    'resnet18-f37072fd.pth',
    'resnet18.npu.json',
    'resnet18.npu.bin',
    'resnet18.validation.npy',
    'resnet18.conversion.json',
    'acceptance.json'
)
foreach ($name in $requiredArtifacts) {
    $path = Join-Path $resolvedArtifacts $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required Vivado artifact is missing: $name"
    }
}
foreach ($name in $requiredModel) {
    $path = Join-Path $resolvedModel $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required model asset is missing: $name"
    }
}

Push-Location $repositoryRoot
try {
    $sourceCommit = (& git rev-parse HEAD).Trim().ToLowerInvariant()
    if ($LASTEXITCODE -ne 0 -or $sourceCommit -notmatch '^[0-9a-f]{40}$') {
        throw 'Cannot determine the exact source commit'
    }
    $artifactManifest = Get-Content -Raw -LiteralPath (
        Join-Path $resolvedArtifacts 'npu_matrix.manifest.json'
    ) | ConvertFrom-Json
    $artifactCommit = ([string]$artifactManifest.source_commit).ToLowerInvariant()
    if (
        $artifactCommit -ne $sourceCommit -and
        -not $AllowArtifactCommitMismatch
    ) {
        throw "Vivado artifacts use $artifactCommit but this checkout uses $sourceCommit; rebuild the overlay from this commit"
    }
    if ($artifactCommit -ne $sourceCommit) {
        Write-Warning "Development-only source mismatch: artifacts=$artifactCommit checkout=$sourceCommit"
    }
    Invoke-CheckedCommand -Command 'python' -Arguments @(
        '-m', 'src.runtime.verify_overlay', $resolvedArtifacts
    )
    Invoke-CheckedCommand -Command 'python' -Arguments @(
        'examples/resnet18/package_example.py',
        '--model-dir', $resolvedModel,
        '--check-only'
    )
}
finally {
    Pop-Location
}

$target = "$BoardUser@$BoardHost"
$remoteBase = $RemoteRoot.TrimEnd('/')
$remoteDeployment = "$remoteBase/releases/$DeploymentId"
Write-Host "Deployment target: $target`:$remoteDeployment"

if ($DryRun) {
    Write-Host 'PASS: deployment inputs verified; no SSH or SCP command executed'
    exit 0
}

Assert-CommandAvailable -Name 'ssh'
Assert-CommandAvailable -Name 'scp'
Invoke-CheckedCommand -Command 'ssh' -Arguments @(
    $target,
    "set -eu; test ! -e '$remoteDeployment'; mkdir -p '$remoteDeployment/examples' '$remoteDeployment/src' '$remoteDeployment/build/vivado/npu_matrix_8x8'"
)
Invoke-CheckedCommand -Command 'scp' -Arguments @(
    '-r', '--',
    (Join-Path $repositoryRoot 'examples/resnet18'),
    "${target}:$remoteDeployment/examples/"
)
foreach ($directory in @('model', 'runtime', 'export')) {
    Invoke-CheckedCommand -Command 'scp' -Arguments @(
        '-r', '--',
        (Join-Path $repositoryRoot "src/$directory"),
        "${target}:$remoteDeployment/src/"
    )
}
Invoke-CheckedCommand -Command 'scp' -Arguments @(
    '-r', '--',
    $resolvedArtifacts,
    "${target}:$remoteDeployment/build/vivado/npu_matrix_8x8/"
)

$mismatchOption = if ($AllowArtifactCommitMismatch) {
    ' --allow-source-mismatch'
} else {
    ''
}
$remoteCommand = "set -eu; cd '$remoteDeployment'; test -r /etc/profile.d/xrt_setup.sh; source /etc/profile.d/xrt_setup.sh; test -r /etc/profile.d/pynq_venv.sh; source /etc/profile.d/pynq_venv.sh; test -x /usr/local/share/pynq-venv/bin/python3; sudo -n XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 examples/resnet18/run_on_board.py --artifact-dir build/vivado/npu_matrix_8x8/artifacts --expected-source-commit '$artifactCommit' --deployed-source-commit '$sourceCommit'$mismatchOption --evidence board-evidence.json"
Invoke-CheckedCommand -Command 'ssh' -Arguments @($target, $remoteCommand)

$evidenceDirectory = Split-Path -Parent $resolvedEvidence
if ($evidenceDirectory) {
    New-Item -ItemType Directory -Force -Path $evidenceDirectory | Out-Null
}
Invoke-CheckedCommand -Command 'scp' -Arguments @(
    '--',
    "${target}:$remoteDeployment/board-evidence.json",
    $resolvedEvidence
)
Write-Host "PASS: physical evidence downloaded to $resolvedEvidence"
