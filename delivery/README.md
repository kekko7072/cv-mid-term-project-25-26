# RAI GROUP [MID-TERM PROJECT CV]

This folder contains the C++/OpenCV delivery package.

## Requirements

- CMake 3.16 or newer
- A C++17 compiler
- OpenCV installed and visible to CMake

## Build

Open a terminal inside `delivery/` and run:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This creates the executable:

```sh
./build/main
```

## Run On The Full Dataset

The easiest way, from inside `delivery/`, is to use the dataset in the parent project folder:

```sh
./build/main --raw ../data/raw --labels ../data/labels --output output
```

Generated files will be saved in:

- `output/results/`
- `output/metrics/`

## Run On One Sequence Only

Example:

```sh
./build/main ../data/raw/frog output/frog
```

This creates:

- `output/frog/0000.txt`
- `output/frog/0000.png`
- `output/frog/moving_points.txt`

## Optional Dataset Layout

If you copy the dataset inside this `delivery/` folder as:

```text
delivery/
  data/
    raw/
    labels/
```

you can also run:

```sh
./build/main --raw data/raw --output output
```

In this case the program automatically uses `data/labels` if it exists.
