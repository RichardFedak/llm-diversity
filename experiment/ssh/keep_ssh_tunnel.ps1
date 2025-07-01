$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$envFile = Join-Path $scriptDir ".env"

# Read the .env file
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*SSH_CONNECTION\s*=\s*["'']?(.+?)["'']?\s*$') {
            $env:SSH_CONNECTION = $matches[1].Trim()
        }
    }
}

if (-not $env:SSH_CONNECTION) {
    Write-Error "SSH_CONNECTION not found or empty in .env file."
    exit 1
}

# Parse SSH command
$sshParts = $env:SSH_CONNECTION -split '\s+'
$sshExe = $sshParts[0]
$sshArgs = $sshParts[1..($sshParts.Length - 1)] -join ' '

Write-Host "Keeping SSH tunnel alive: $env:SSH_CONNECTION"

while ($true) {
    try {
        Write-Host "Starting SSH tunnel..."
        Start-Process -NoNewWindow -Wait -FilePath $sshExe -ArgumentList $sshArgs
        Write-Warning "SSH tunnel exited. Reconnecting in 5 seconds..."
    } catch {
        Write-Error "SSH process failed: $_"
    }
    Start-Sleep -Seconds 5
}
