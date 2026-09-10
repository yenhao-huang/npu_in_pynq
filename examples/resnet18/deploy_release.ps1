[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$DeploymentId,

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$BoardHost = '192.168.2.99',

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$BoardUser = 'xilinx',

    [ValidatePattern('^/[A-Za-z0-9._/-]+$')]
    [string]$RemoteRoot = '/home/xilinx/jupyter_notebooks/npu_resnet18',

    [string]$ArtifactDir = 'build/vivado/npu_matrix_8x8/artifacts',

    [switch]$AllowArtifactCommitMismatch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "Required deployment command is unavailable: $Command"
    }
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$resolvedArtifacts = (Resolve-Path (
    Join-Path $repositoryRoot $ArtifactDir
)).Path
$artifactManifestPath = Join-Path (
    $resolvedArtifacts
) 'npu_matrix.manifest.json'
$artifactManifest = Get-Content -Raw -LiteralPath (
    $artifactManifestPath
) | ConvertFrom-Json
$artifactCommit = ([string]$artifactManifest.source_commit).ToLowerInvariant()

Push-Location $repositoryRoot
try {
    Invoke-CheckedCommand -Command 'python' -Arguments @(
        'examples/resnet18/scripts/verify_demo.py',
        '--artifact-dir', $resolvedArtifacts
    )
    $sourceCommit = (& git rev-parse HEAD).Trim().ToLowerInvariant()
    if ($LASTEXITCODE -ne 0) {
        throw 'Cannot determine the deployed source commit'
    }
    if ($artifactCommit -ne $sourceCommit -and -not $AllowArtifactCommitMismatch) {
        throw 'Artifact and deployed commits differ; explicitly allow the development overlay or rebuild.'
    }
}
finally {
    Pop-Location
}

$target = "$BoardUser@$BoardHost"
$remoteBase = $RemoteRoot.TrimEnd('/')
$remoteDeployment = "$remoteBase/releases/$DeploymentId"
$metadataPath = Join-Path (
    [IO.Path]::GetTempPath()
) "npu-resnet18-$([Guid]::NewGuid().ToString('N')).json"
$metadata = [ordered]@{
    allow_source_mismatch = [bool]$AllowArtifactCommitMismatch
    array_size = 8
    artifact_source_commit = $artifactCommit
    deployed_source_commit = $sourceCommit
    deployment_id = $DeploymentId
    format = [ordered]@{ major = 1; minor = 0 }
    magic = 'NPU_RESNET18_DEPLOYMENT'
}
[IO.File]::WriteAllText(
    $metadataPath,
    (($metadata | ConvertTo-Json -Compress) + "`n"),
    [Text.UTF8Encoding]::new($false)
)

Write-Host "Deployment target: $target`:$remoteDeployment"
try {
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
    Invoke-CheckedCommand -Command 'scp' -Arguments @(
        '--',
        $metadataPath,
        "${target}:$remoteDeployment/deployment.json"
    )
}
finally {
    Remove-Item -LiteralPath $metadataPath -Force -ErrorAction SilentlyContinue
}

Write-Host "PASS: files copied to $target`:$remoteDeployment"
Write-Host "NEXT: open examples/resnet18/resnet18.ipynb on the PYNQ-Z1"
Write-Host 'Alternative: run examples/resnet18/run_on_board.py manually on the PYNQ-Z1'
