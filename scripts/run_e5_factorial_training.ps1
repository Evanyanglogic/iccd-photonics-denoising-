$ErrorActionPreference = "Stop"

$decision = Get-Content -Raw "reports\e5_noise_factorial\decoupling_decision.json" | ConvertFrom-Json
if ($decision.status -ne "GO_TO_TRAIN") {
    throw "Factorial data did not pass decoupling validation. Training is blocked."
}

$variants = @("P-L", "P-H", "H-L", "H-H")
$seeds = @(20260716, 20260717, 20260718)

foreach ($variant in $variants) {
    foreach ($seed in $seeds) {
        $runName = "${variant}_seed${seed}"
        $runDir = "reports\e5_noise_factorial\training\${runName}"
        python scripts\train_manifest_denoiser_baseline.py `
            --experiment-id "e5_${runName}" `
            --model-type residual_small `
            --channels 16 `
            --depth 3 `
            --train-pairs "reports\e5_noise_factorial\${variant}\pairs.csv" `
            --train-splits "reports\e5_noise_factorial\${variant}\splits.yaml" `
            --val-pairs "reports\e5_noise_factorial\${variant}\pairs.csv" `
            --val-splits "reports\e5_noise_factorial\${variant}\splits.yaml" `
            --epochs 100 `
            --batch-size 4 `
            --patch-size 128 `
            --val-patch-size 256 `
            --lr 0.001 `
            --weight-decay 0 `
            --seed $seed `
            --device cuda `
            --output-dir $runDir

        python scripts\evaluate_factorial_probe.py `
            --config configs\e5_noise_factorial.yaml `
            --checkpoint "${runDir}\checkpoints\best.pth" `
            --variant $variant `
            --seed $seed `
            --device cuda `
            --output-dir "${runDir}\real_eval"
    }
}
