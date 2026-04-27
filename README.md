# RAI GROUP [MID-TERM PROJECT CV]

## Build

```sh
sh macos_compile.sh
```

## Run End to End

```sh
./build/main --raw data/raw --output output
```

The motion-handoff output tree is:

- `output/motion/sequence_manifest.csv`
- `output/motion/sequence_summary.csv`
- `output/motion/tracks/`
- `output/motion/debug/`
- `output/motion/review_examples/`
- `output/motion/notes/`

`--labels` is optional and is only used to link label files in the sequence manifest. Localization is intentionally left as TODO for Member 2.
