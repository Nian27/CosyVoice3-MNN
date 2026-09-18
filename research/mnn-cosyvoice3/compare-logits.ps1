# compare-logits.ps1
#
# Clock-independent numerical A/B for MNN's ATTENTION_OPTION (KV quantization).
#
# Why this exists: on a phone whose CPU clocks may be clamped, wall-clock A/B
# cannot resolve anything (see docs/MNN_HINT_AB_2026-09-15.md §降频钳制), and
# comparing sampled tokens is insensitive because greedy decoding collapses this
# model to a single repeated token. Comparing the raw prefill/decode logits
# sidesteps both problems: it is deterministic and needs no clock.
#
# Prerequisite: MNN built with the MNN_LOGITS_DUMP instrumentation in
# transformers/llm/engine/src/llm.cpp (Llm::sample). Run the LLM benchmark once
# per attention_mode with that env var set, pull each directory, then run this.
#
# Example:
#   .\compare-logits.ps1 -Reference m8a -Candidates m8b,m9,m10,m11,m13 -Root $env:TEMP\logits-ab

param(
    [string]$Reference = 'm8a',
    [string[]]$Candidates = @('m8b', 'm9', 'm10', 'm11', 'm13'),
    [string]$Root = (Join-Path $env:TEMP 'logits-ab'),
    [int[]]$Steps = @(0, 1, 7)
)

$ErrorActionPreference = 'Stop'

Add-Type -TypeDefinition @'
using System;using System.IO;
public static class LogitsCmp {
  public static float[] Load(string p){var b=File.ReadAllBytes(p);var f=new float[b.Length/4];Buffer.BlockCopy(b,0,f,0,b.Length);return f;}
  public static string Run(string a,string b){
    var x=Load(a);var y=Load(b);
    double maxabs=0,sumsq=0,dot=0,nx=0,ny=0;int am=0,bm=0;
    for(int i=0;i<x.Length;i++){
      double d=Math.Abs((double)x[i]-y[i]); if(d>maxabs)maxabs=d; sumsq+=(double)(x[i]-y[i])*(x[i]-y[i]);
      dot+=(double)x[i]*y[i]; nx+=(double)x[i]*x[i]; ny+=(double)y[i]*y[i];
      if(x[i]>x[am])am=i; if(y[i]>y[bm])bm=i;
    }
    double rms=Math.Sqrt(sumsq/x.Length); double cos=dot/(Math.Sqrt(nx)*Math.Sqrt(ny));
    return string.Format("maxabs={0:E3} rms={1:E3} cosine={2:F8} argmax {3}/{4} {5}",
                         maxabs, rms, cos, am, bm, am==bm?"SAME":"DIFF");
  }
}
'@

Write-Host "reference = $Reference"
foreach ($k in $Candidates) {
    Write-Host "--- $k"
    foreach ($s in $Steps) {
        $a = Join-Path $Root "$Reference\logits-step$s.bin"
        $b = Join-Path $Root "$k\logits-step$s.bin"
        if ((Test-Path $a) -and (Test-Path $b)) {
            Write-Host ("   step{0}: {1}" -f $s, [LogitsCmp]::Run($a, $b))
        } else {
            Write-Host ("   step{0}: MISSING input" -f $s)
        }
    }
}
