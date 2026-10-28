# one-hot-encoding

Converts categorical table columns into one-hot encoded numeric features.

## Runtime inputs

Supply input files according to the schemas in `app.yml`.

## Inputs and outputs

### Inputs

| Name | Type | Description |
| --- | --- | --- |
| `features` | CSV | Input CSV table. Optional first column named 'id' (kept as identifier). Other columns can be numeric or categorical. |

### Outputs

| Name | Type | Description |
| --- | --- | --- |
| `encoded` | CSV | One-hot encoded CSV (same rows; additional dummy columns). |

## Configuration

| Name | Default | Description |
| --- | --- | --- |
| `columns` | `` | Comma-separated list of columns to one-hot encode. If empty, all object/category/bool columns are encoded. |
| `drop_first` | `0` | If true, drop the first category per feature (k-1 dummies). |
| `bool_to_number` | `1` | If true, convert boolean columns to numeric 0/1 instead of one-hot encoding them. |
| `dummy_na` | `0` | If true, create a dedicated dummy column for NaN values in encoded columns. |
| `handle_unknown` | `ignore` | How to handle unknown categories. Kept for consistency; pandas.get_dummies encodes categories present in the input. |
| `prefix_sep` | `__` | Separator between column name and category in generated dummy column names. |
| `keep_original` | `1` | If true, keep original categorical columns in addition to dummy columns. |

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# Set APP_ID and platform URLs before connecting.
python main.py
```

The Docker base image provides the FL-Net API. `requirements.txt` lists additional
runtime dependencies; `requirements-dev.txt` includes the API for local development.
Runtime datasets and trained models must be supplied separately; see `data/README.md`.

Build with `docker build -t tool-one-hot-encoding .`.
