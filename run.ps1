<#
.SYNOPSIS
    MANET coverage demo – egylépéses indítás Windows-on.

.DESCRIPTION
    1. Python (>= 3.10) keresése; ha nincs, telepítés (winget, vagy python.org installer).
    2. Virtuális környezet (.venv) létrehozása / ellenőrzése.
    3. Python csomagok (requirements.txt) ellenőrzése, hiány esetén telepítése.
    4. Gyors importteszt a projekt moduljain.
    5. Streamlit szerver indítása, majd a demó megnyitása az alapértelmezett böngészőben.
    Leállítás: Ctrl+C ebben az ablakban.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run.ps1
    .\run.ps1 -Port 8600 -NoBrowser
    .\run.ps1 -SetupOnly
#>
param(
    [int]$Port = 8501,          # kívánt port; ha foglalt, a következő szabadot választja
    [switch]$NoBrowser,         # ne nyissa meg a böngészőt
    [switch]$SetupOnly          # csak ellenőrzés/telepítés, indítás nélkül
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest PS 5.1-ben progress bar nélkül sokkal gyorsabb
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
Set-Location -LiteralPath $PSScriptRoot

$MinPython   = [version]'3.10'
$PyWinget    = 'Python.Python.3.12'
$PyInstaller = 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe'
$VenvDir     = Join-Path $PSScriptRoot '.venv'
$VenvPy      = Join-Path $VenvDir 'Scripts\python.exe'
$ReqFile     = Join-Path $PSScriptRoot 'requirements.txt'
$ReqStamp    = Join-Path $VenvDir 'requirements.sha256'
$TotalSteps  = if ($SetupOnly) { 4 } else { 5 }
$script:StepNo = 0

# --------------------------------------------------------------------------- logging

function Write-Log([string]$Level, [string]$Msg) {
    $ts = Get-Date -Format 'HH:mm:ss'
    $color = @{ INFO = 'Gray'; OK = 'Green'; WARN = 'Yellow'; ERR = 'Red' }[$Level]
    Write-Host "[$ts] " -NoNewline -ForegroundColor DarkGray
    Write-Host ("{0,-4} " -f $Level) -NoNewline -ForegroundColor $color
    Write-Host $Msg
}
function Info([string]$m) { Write-Log 'INFO' $m }
function Ok([string]$m)   { Write-Log 'OK' $m }
function Warn([string]$m) { Write-Log 'WARN' $m }
function Fail([string]$m) { throw $m }   # a fő catch blokk naplózza

function Step([string]$Title) {
    $script:StepNo++
    Write-Host ''
    Write-Host ("=== [{0}/{1}] {2} " -f $script:StepNo, $TotalSteps, $Title).PadRight(78, '=') -ForegroundColor Cyan
}

# --------------------------------------------------------------------------- helpers

function Update-SessionPath {
    # Telepítés után az új PATH csak új folyamatokban látszik – újraolvassuk a registryből.
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = (@($machine, $user) | Where-Object { $_ }) -join ';'
}

function Get-PythonInfo([string]$Exe, [string[]]$PreArgs = @()) {
    # Visszaadja @{ Path; Version }-t, ha az $Exe egy működő Python; különben $null.
    # (A Microsoft Store "python.exe" álparancs nem ad kimenetet -> kiszűrődik.)
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $Exe @PreArgs -c "import sys; print(sys.executable); print('%d.%d.%d' % sys.version_info[:3])" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $out -or @($out).Count -lt 2) { return $null }
        return @{ Path = @($out)[0].Trim(); Version = [version]@($out)[1].Trim() }
    } catch { return $null }
}

function Find-Python {
    $candidates = @(
        @{ Exe = 'py';      Args = @('-3') },
        @{ Exe = 'python';  Args = @() },
        @{ Exe = 'python3'; Args = @() }
    )
    foreach ($dir in @("$env:LOCALAPPDATA\Programs\Python", "$env:ProgramFiles")) {
        if (Test-Path $dir) {
            Get-ChildItem -Path $dir -Directory -Filter 'Python3*' -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending |
                ForEach-Object { $candidates += @{ Exe = (Join-Path $_.FullName 'python.exe'); Args = @() } }
        }
    }
    foreach ($c in $candidates) {
        if (-not (Get-Command $c.Exe -ErrorAction SilentlyContinue)) { continue }
        $info = Get-PythonInfo $c.Exe $c.Args
        if ($null -eq $info) { continue }
        if ($info.Version -ge $MinPython) { return $info }
        Warn "Talált Python túl régi: $($info.Version) ($($info.Path)), kell >= $MinPython"
    }
    return $null
}

function Install-Python {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Info "Telepítés winget-tel: $PyWinget (felhasználói szintre, rendszergazda nem kell)..."
        & winget install --exact --id $PyWinget --scope user --silent `
            --accept-package-agreements --accept-source-agreements --disable-interactivity
        if ($LASTEXITCODE -eq 0) { Ok 'winget telepítés kész.'; return }
        Warn "winget hibakód: $LASTEXITCODE – áttérés a python.org installerre."
    } else {
        Info 'winget nem elérhető – python.org installer használata.'
    }
    $tmp = Join-Path $env:TEMP 'manet-python-installer.exe'
    Info "Letöltés: $PyInstaller"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $PyInstaller -OutFile $tmp -UseBasicParsing
    Info 'Csendes telepítés (felhasználói szint, PATH-ba felvéve)...'
    $p = Start-Process -FilePath $tmp -Wait -PassThru `
        -ArgumentList '/quiet', 'InstallAllUsers=0', 'PrependPath=1', 'Include_launcher=1', 'Include_test=0'
    Remove-Item $tmp -ErrorAction SilentlyContinue
    if ($p.ExitCode -ne 0) { Fail "A Python installer hibakóddal lépett ki: $($p.ExitCode)" }
    Ok 'Python installer kész.'
}

function Test-Import([string]$Module) {
    # PS 5.1: Stop mellett a natív stderr átirányítása kivételt dobna -> helyben Continue.
    $ErrorActionPreference = 'Continue'
    & $VenvPy -c "import $Module" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Get-FileSha256([string]$File) {
    (Get-FileHash -Algorithm SHA256 -LiteralPath $File).Hash
}

function Test-PortFree([int]$P) {
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $P)
        $listener.Start()
        return $true
    } catch { return $false } finally { if ($listener) { $listener.Stop() } }
}

# --------------------------------------------------------------------------- main

Write-Host ''
Write-Host 'MANET coverage demo – indító script' -ForegroundColor White
Info "Projektmappa: $PSScriptRoot"
$sw = [Diagnostics.Stopwatch]::StartNew()

try {
    # ---- 1. Python ------------------------------------------------------------------------------
    Step 'Python ellenőrzése'
    $py = Find-Python
    if ($null -eq $py) {
        Warn "Nem található Python >= $MinPython."
        Install-Python
        Update-SessionPath
        $py = Find-Python
        if ($null -eq $py) {
            Fail 'A Python telepítése után sem található. Nyiss új PowerShell ablakot és futtasd újra a scriptet.'
        }
    }
    Ok "Python $($py.Version): $($py.Path)"

    # ---- 2. venv --------------------------------------------------------------------------------
    Step 'Virtuális környezet (.venv)'
    $venvOk = $false
    if (Test-Path -LiteralPath $VenvPy) {
        $vinfo = Get-PythonInfo $VenvPy
        if ($null -ne $vinfo -and $vinfo.Version -ge $MinPython) {
            $venvOk = $true
            Ok "Meglévő .venv működik (Python $($vinfo.Version))."
        } else {
            Warn 'A meglévő .venv sérült vagy más gépről származik – újralétrehozás.'
            Remove-Item -LiteralPath $VenvDir -Recurse -Force
        }
    }
    if (-not $venvOk) {
        Info 'venv létrehozása...'
        & $py.Path -m venv $VenvDir
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPy)) { Fail 'A venv létrehozása sikertelen.' }
        Ok '.venv létrehozva.'
    }

    # ---- 3. csomagok ----------------------------------------------------------------------------
    Step 'Python csomagok (requirements.txt)'
    $reqHash = Get-FileSha256 $ReqFile
    $stampOk = (Test-Path -LiteralPath $ReqStamp) -and ((Get-Content -LiteralPath $ReqStamp -Raw).Trim() -eq $reqHash)
    $missing = @()
    foreach ($mod in 'numpy', 'networkx', 'matplotlib', 'pandas', 'streamlit', 'PIL', 'pytest') {
        if (-not (Test-Import $mod)) { $missing += $mod }
    }
    if ($missing.Count -eq 0 -and $stampOk) {
        Ok 'Minden csomag telepítve, a requirements.txt nem változott.'
    } else {
        if ($missing.Count -gt 0) { Info "Hiányzó csomagok: $($missing -join ', ')" }
        else { Info 'A requirements.txt változott – frissítés.' }
        Info 'pip frissítése...'
        & $VenvPy -m pip install --upgrade pip --disable-pip-version-check --progress-bar off
        if ($LASTEXITCODE -ne 0) { Warn 'A pip frissítése nem sikerült – folytatás a meglévővel.' }
        Info 'Csomagok telepítése (első alkalommal pár perc is lehet)...'
        & $VenvPy -m pip install -r $ReqFile --disable-pip-version-check --progress-bar off
        if ($LASTEXITCODE -ne 0) { Fail 'A csomagok telepítése sikertelen (internetkapcsolat?).' }
        Set-Content -LiteralPath $ReqStamp -Value $reqHash -Encoding ascii
        Ok 'Csomagok telepítve.'
    }

    # ---- 4. importteszt -------------------------------------------------------------------------
    Step 'Projekt ellenőrzése (importteszt)'
    & $VenvPy -c "import manet.world, manet.sim, manet.render, manet.summary; from manet.algorithms import ALGORITHM_NAMES, get_algorithm; [get_algorithm(n) for n in ALGORITHM_NAMES]; import streamlit; print('algoritmusok:', ', '.join(ALGORITHM_NAMES), '| streamlit', streamlit.__version__)"
    if ($LASTEXITCODE -ne 0) { Fail 'A projekt moduljai nem importálhatók.' }
    Ok 'Projekt rendben.'

    if ($SetupOnly) {
        Write-Host ''
        Ok ("Telepítés kész ({0:N1} s). Indítás: .\run.ps1" -f $sw.Elapsed.TotalSeconds)
        exit 0
    }

    # ---- 5. indítás -----------------------------------------------------------------------------
    Step 'Alkalmazás indítása'
    $p = $Port
    while (-not (Test-PortFree $p)) {
        Warn "A $p port foglalt."
        $p++
        if ($p -gt $Port + 20) { Fail 'Nem találtam szabad portot.' }
    }
    $url = "http://localhost:$p"
    Info "Streamlit szerver indítása: $url"
    $server = Start-Process -FilePath $VenvPy -NoNewWindow -PassThru -ArgumentList @(
        '-m', 'streamlit', 'run', 'app.py',
        '--server.headless', 'true',
        '--server.port', "$p",
        '--browser.gatherUsageStats', 'false'
    )

    Info 'Várakozás, amíg a szerver válaszol...'
    $ready = $false
    for ($i = 0; $i -lt 120; $i++) {
        if ($server.HasExited) { Fail "A szerver leállt (kilépési kód: $($server.ExitCode))." }
        try {
            $r = Invoke-WebRequest -Uri "$url/_stcore/health" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { $ready = $true; break }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { Fail 'A szerver 60 s alatt sem indult el.' }
    Ok ("A demó fut: {0}  (indítás {1:N1} s alatt)" -f $url, $sw.Elapsed.TotalSeconds)

    if ($NoBrowser) {
        Info 'Böngésző megnyitása kihagyva (-NoBrowser).'
    } else {
        Info 'Megnyitás az alapértelmezett böngészőben...'
        Start-Process $url
    }
    Write-Host ''
    Write-Host 'Leállítás: Ctrl+C (vagy zárd be ezt az ablakot).' -ForegroundColor Yellow
    Write-Host ''

    try {
        $server.WaitForExit()
    } finally {
        if (-not $server.HasExited) {
            Info 'Szerver leállítása...'
            Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
        }
        Info 'Viszlát!'
    }
} catch {
    Write-Host ''
    Write-Log 'ERR' "Hiba: $($_.Exception.Message)"
    Write-Log 'ERR' 'A fenti naplóból látszik, melyik lépés akadt el.'
    exit 1
}
