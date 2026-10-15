# Launch-method comparison for StudentAIAssistantWeb.exe.
# Runs a controlled idle lifecycle trial for BOTH:
#   A) PowerShell Start-Process   B) Python subprocess.Popen (same binary/config)
# Each trial: launch -> poll health every 0.5s for T seconds -> record exit if any.
# When the trial ends normally, the harness kills ONLY its own tracked PID.
# Results are written to JSON. No secrets are logged.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\launch_compare.ps1
$ErrorActionPreference = "Stop"

$exe = "D:\chatbot ai\dist\StudentAIAssistantWeb\StudentAIAssistantWeb.exe"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$out = Join-Path $scriptDir "..\reports\launch_compare.json"
$trials = 3
$holdSeconds = 20

$results = @()

function Test-Port($port) {
    $client = New-Object Net.Sockets.TcpClient
    try {
        $iar = $client.BeginConnect("127.0.0.1", $port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(1500) -and $client.Connected
        if ($ok) { return $true }
        return $false
    } catch { return $false }
    finally { $client.Dispose() }
}

# ---------------- Method A: Start-Process ----------------
for ($i = 0; $i -lt $trials; $i++) {
    $port = 8040 + $i
    $dataDir = Join-Path $env:TEMP ("LaunchA_" + $i)
    New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
    $env:STUDENTAI_DATA_DIR = $dataDir
    $outF = Join-Path $dataDir "stdout.log"
    $errF = Join-Path $dataDir "stderr.log"
    $p = Start-Process -FilePath $exe -ArgumentList @("--no-browser", "--port", "$port") `
        -WorkingDirectory $dataDir -RedirectStandardOutput $outF -RedirectStandardError $errF -PassThru
    $pid_ = $p.Id
    Remove-Item Env:\STUDENTAI_DATA_DIR

    $healthSeen = $false
    $start = Get-Date
    $exitedEarly = $false
    $exitCode = $null
    while (((Get-Date) - $start).TotalSeconds -lt $holdSeconds) {
        if ($p.HasExited) {
            $exitedEarly = $true
            $exitCode = $p.ExitCode
            break
        }
        if (Test-Port $port) { $healthSeen = $true }
        Start-Sleep -Milliseconds 500
    }
    $took = [math]::Round(((Get-Date) - $start).TotalSeconds, 2)
    if (-not $p.HasExited) {
        Stop-Process -Id $pid_ -Force
        $p.WaitForExit()
    }
    $stderrTail = ""
    if (Test-Path $errF) {
        $stderrTail = (Get-Content $errF -Tail 8 -ErrorAction SilentlyContinue) -join "`n"
    }
    $results += [pscustomobject]@{
        method = "Start-Process"
        trial = $i
        pid = $pid_
        stayed_alive = (-not $exitedEarly)
        exited_early_s = $(if ($exitedEarly) { $took } else { $null })
        exit_code = $(if ($exitCode -ne $null) { $exitCode } else { "n/a" })
        health_seen = $healthSeen
        stderr_tail = $stderrTail
    }
    Write-Output ("A trial $i pid=$pid_ alive=$(-not $exitedEarly) health=$healthSeen exit=" + $(if ($exitCode -ne $null) { $exitCode } else { "n/a" }))
}

# ---------------- Method B: subprocess.Popen ----------------
# Call python which uses subprocess.Popen with identical args + STUDENTAI_DATA_DIR.
$pyProbe = Join-Path $scriptDir "launch_compare_popen.py"
$pyOut = & python $pyProbe $exe $trials $holdSeconds

$methodB = $null
foreach ($line in $pyOut) {
    if ($line -like "RESULT_JSON=*") {
        $json = $line.Substring("RESULT_JSON=".Length)
        $methodB = $json | ConvertFrom-Json
    } else {
        Write-Output $line
    }
}
if ($methodB -ne $null) {
    foreach ($r in $methodB) { $results += $r }
}

$results | ConvertTo-Json -Depth 4 | Out-File -FilePath $out -Encoding utf8
Write-Output ("JSON written to: " + $out)