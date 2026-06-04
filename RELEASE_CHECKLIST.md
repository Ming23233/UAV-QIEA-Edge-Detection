# GitHub Release Checklist

Before making the repository public:

- Replace the placeholder repository URL in `CITATION.cff`.
- Confirm whether the final accepted paper title should be updated in `README.md`.
- Do not upload raw datasets, checkpoints, pretrained weights, local training logs, large detection-output JSON files, or manuscript drafts.
- Keep only small manuscript-level result artifacts in `results/`.
- Verify all scripts use public or relative defaults, or document required command-line paths.
- Check third-party licenses under `third_party/ByteTrack/`.
- If a journal requires checkpoints, upload them to a separate release or storage location only after confirming license and file-size constraints.

Do not add:

```text
*.pth
*.pt
stdout.log
stderr.log
train_status.log
last_ckpt.pth
best_ckpt.pth
data/
outputs/
```
