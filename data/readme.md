# Dataset Availability

The dataset used in this project is derived from the CERN ATLAS TileCal simulation pipeline.

This data is not included in the repository due to distribution and usage restrictions.

## Accessing the Data

To reproduce the results, users will need access to the original dataset. Please refer to official CERN ATLAS resources or consult project mentors/maintainers for guidance on obtaining access.

## Usage

The codebase is designed to work with locally available data. Update the data path accordingly, for example:

```python
data = torch.load("path/to/your/shard.pt")
```

## Scope

This repository focuses on:

- Model implementation
- Training and evaluation pipelines
- Benchmarking results

No restricted or proprietary data is distributed within this repository. 