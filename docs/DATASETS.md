# Dataset Notes

This repository does not redistribute datasets.

## Primary Dataset

The primary experiments use VisDrone-DET converted to COCO format. The expected layout is:

```text
data/visdrone_det_coco/
|-- annotations/
|   |-- instances_train.json
|   |-- instances_val.json
|   |-- search_train.json
|   `-- search_val.json
`-- images/
```

Pass the dataset path using `--dataset-root`.

The main full-training experiments use `instances_train.json` and `instances_val.json`. The proxy search uses `search_train.json` and `search_val.json`, which can be generated from the converted VisDrone COCO annotations with the repository scripts.

## External Engineering Case

The current manuscript also reports an AU-AIR engineering case in COCO format with zero-shot evaluation and 30% target-domain fine-tuning. AU-AIR data are not redistributed in this repository; prepare the dataset locally and follow the script arguments in `scripts/run_stage24_auair_engineering_case.py`.

## Conversion Scripts

Useful conversion scripts:

- `third_party/ByteTrack/tools/convert_visdrone_det_to_coco.py`
- `third_party/ByteTrack/tools/convert_uavdt_to_coco.py`
- `scripts/run_stage22_dronevehicle_external_case.py`
- `scripts/run_stage24_auair_engineering_case.py`

Check the license and terms of each dataset before use.
