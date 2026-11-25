# Your own floor plans

Maps made with `scripts/import_floorplan.py` are saved here (a PNG and a YAML each), for example:

```bash
python scripts/import_floorplan.py level2.png --name level2 --width-m 42
python scripts/demo.py --scenario maps/level2.yaml
```

The hospital, office and warehouse maps are not here: `scripts/get_maps.sh` downloads them into
`third_party/floorplans/`.
