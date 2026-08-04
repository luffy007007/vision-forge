# CV Lab

<!-- Notes only for now. Full README comes at the end of the build. -->

## Classifier results

`convnext_tiny` (timm, ImageNet-pretrained) fine-tuned on EuroSAT RGB — 27,000
64×64 Sentinel-2 tiles, 10 land-use classes, 80/20 split (seed 0). Trained on
Colab, one T4, 5 epochs per run.

| Run | What was trainable | LR | Val accuracy |
|-----|--------------------|-----|--------------|
| A | classifier head only, backbone frozen | 1e-3 | **0.9530** |
| B | + last ConvNeXt stage unfrozen | 1e-4 | **0.9719** |

**The delta is +1.9 points.** Worth reading carefully: a frozen backbone with
only a linear head trained already reaches ~95%. The ImageNet features transfer
to overhead satellite imagery almost intact, and unfreezing one stage buys
under two points on top of that.

Epoch-to-epoch variation within each run was ~0.3%, so the gain is real but
small. Both numbers are the final epoch, not the best one — Run A actually
peaked at 0.9557 on epoch 2.

Two runs, then stop. Chasing another point of accuracy was explicitly out of
scope for this project.

## Class order

Saved inside `models/classifier.pt` as `ckpt["classes"]`, not hardcoded
anywhere — the argmax index means nothing without it:

```
AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
Pasture, PermanentCrop, Residential, River, SeaLake
```

Checkpoint layout: `{"model": str, "classes": list[str], "state_dict": ...}`.

## TODO for the full README

- what the project is, screenshot of the Streamlit app
- setup + run instructions
- the `CVResult` contract and why every module shares it
- `assets/opencv_grid.png` (Phase 1 operations)
- example `scene_report.json` from the full pipeline
- licensing: Ultralytics YOLO is **AGPL-3.0** — fine here, note RF-DETR as the
  Apache-2.0 alternative. SAM and CLIP both load via HF `transformers`.
  SigLIP is CLIP's stronger successor, worth a mention.
- `timm` for models, `torchvision` for transforms and dataset loaders only
