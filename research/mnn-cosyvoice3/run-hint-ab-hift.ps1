# run-hint-ab-hift.ps1
#
# Single-variable A/B sweep of MNN runtime hints on the HiFT core model.
#
# Why min-of-N: the SM8850 is a phone, not a benchmark rig. Run-to-run spread on
# a single process is +/-5-20% (thermal drift + heterogeneous-core scheduling),
# which is larger than most of the effects being tested. Variants are therefore
# interleaved round-robin across repeats and summarised by MIN of the per-run
# medians -- the minimum approximates un-throttled capability, whereas the mean
# mostly measures how hot the phone got.
#
# Requires the benchmark built with CosyVoiceHintKnobs.hpp and pushed to the
# device (see research/mnn-cosyvoice3/build-stage2-android.ps1).
#
# Example:
#   .\run-hint-ab-hift.ps1 -Repeats 5 -Loops 10

param(
    [string]$Adb = 'C:\Users\Administrator\AppData\Local\Android\Sdk\platform-tools\adb.exe',
    [string]$DeviceDir = '/data/local/tmp/cosy-mnn-stage2',
    [int]$Frames = 174,
    [int]$Threads = 6,
    [int]$Loops = 10,
    [int]$Repeats = 5,
    [string]$OutFile = 'hint-ab-results.jsonl'
)

$ErrorActionPreference = 'Stop'

# Each entry changes exactly one variable relative to 'base'.
$Variants = [ordered]@{
    'base'          = @{ env = ''; threads = $Threads }
    'mem_high'      = @{ env = 'MNN_HINT_MEMORY=high '; threads = $Threads }
    'kleidiai'      = @{ env = 'MNN_HINT_KLEIDIAI=1 '; threads = $Threads }
    'winograd_1'    = @{ env = 'MNN_HINT_WINOGRAD=1 '; threads = $Threads }
    'littlecore100' = @{ env = 'MNN_HINT_LITTLECORE=100 '; threads = $Threads }
    'prime_first'   = @{ env = 'MNN_HINT_CORE_IDS=6,7,0,1,2,3 '; threads = $Threads }
    'prime_plus_lc' = @{ env = 'MNN_HINT_LITTLECORE=100 MNN_HINT_CORE_IDS=6,7,0,1,2,3 '; threads = $Threads }
    'threads_8'     = @{ env = ''; threads = 8 }
}

if (Test-Path $OutFile) { Remove-Item $OutFile -Force }
$records = New-Object System.Collections.Generic.List[object]

for ($rep = 1; $rep -le $Repeats; $rep++) {
    foreach ($name in $Variants.Keys) {
        $v = $Variants[$name]
        $cmd = "cd $DeviceDir && $($v.env)./CosyVoiceHiFTBenchmark.out hift-core.mnn core inputs " +
               "$DeviceDir/ab-$name.bin $Frames cpu low $Loops $($v.threads)"
        $raw = & $Adb shell $cmd 2>&1 | Out-String
        $line = ($raw -split "`n" | Where-Object { $_ -match '^mode=' } | Select-Object -First 1)
        if (-not $line) {
            Write-Warning "[$name rep$rep] no result line"
            continue
        }
        $record = [ordered]@{
            variant     = $name
            repeat      = $rep
            threads     = $v.threads
            env         = $v.env.Trim()
            medianMs    = [double]([regex]::Match($line, 'median_ms=([\d.]+)').Groups[1].Value)
            firstMs     = [double]([regex]::Match($line, 'first_ms=([\d.]+)').Groups[1].Value)
            memoryMb    = [double]([regex]::Match($line, 'memory_mb=([\d.]+)').Groups[1].Value)
            finite      = [regex]::Match($line, 'finite=(\d)').Groups[1].Value
            memoryMode  = [regex]::Match($line, 'memory_mode=(\S+)').Groups[1].Value
            hints       = [regex]::Match($line, 'hints=(\S+)').Groups[1].Value
        }
        $records.Add([pscustomobject]$record)
        ($record | ConvertTo-Json -Compress) | Add-Content -Path $OutFile -Encoding UTF8
        Write-Host ("[{0,-14} rep{1}] median={2,8:N1} ms  first={3,8:N1} ms  finite={4}" -f `
            $name, $rep, $record['medianMs'], $record['firstMs'], $record['finite'])
    }
}

Write-Host ''
Write-Host '=== summary (min / median of per-run medians) ==='
$summary = $records | Group-Object variant | ForEach-Object {
    $values = $_.Group | ForEach-Object { $_.medianMs } | Sort-Object
    [pscustomobject]@{
        variant   = $_.Name
        n         = $values.Count
        minMs     = [math]::Round($values[0], 1)
        medianMs  = [math]::Round($values[[int]($values.Count / 2)], 1)
        maxMs     = [math]::Round($values[-1], 1)
        spreadPct = [math]::Round(100.0 * ($values[-1] - $values[0]) / $values[0], 1)
    }
} | Sort-Object minMs

$base = ($summary | Where-Object { $_.variant -eq 'base' }).minMs
$summary | ForEach-Object {
    $_ | Add-Member -NotePropertyName 'vsBasePct' -NotePropertyValue (
        if ($base) { [math]::Round(100.0 * ($_.minMs - $base) / $base, 1) } else { 0 }
    ) -Force
    $_
} | Format-Table -AutoSize
