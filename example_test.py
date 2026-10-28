from pathlib import Path

import pandas as pd


def one_hot_encode_csv(
        input_csv: str | Path,
        output_csv: str | Path,
        *,
        columns: str = "",
        # "" => auto encode all categorical (object/category + bool if bool_to_number=False)
        drop_first: bool = False,
        dummy_na: bool = False,
        prefix_sep: str = "__",
        keep_original: bool = False,
        bool_to_number: bool = True,
) -> Path:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    df = pd.read_csv(input_csv)

    # Split id column if present
    ids = None
    if df.shape[1] > 0 and str(df.columns[0]).strip().lower() == "id":
        ids = df.iloc[:, 0]
        feats = df.iloc[:, 1:]
    else:
        feats = df

    # Determine columns to encode
    raw = (columns or "").strip()
    if raw:
        cols_to_encode = [c.strip() for c in raw.split(",") if c.strip()]
        missing = [c for c in cols_to_encode if c not in feats.columns]
        if missing:
            raise ValueError(f"Requested columns not found: {missing}")
    else:
        cols_to_encode = [
            c for c in feats.columns
            if (
                    pd.api.types.is_object_dtype(feats[c])
                    or pd.api.types.is_categorical_dtype(feats[c])
                    or (not bool_to_number and pd.api.types.is_bool_dtype(feats[c]))
            )
        ]

    # Encode
    if not cols_to_encode:
        encoded = feats.copy()
    else:
        if keep_original:
            dummies = pd.get_dummies(
                feats[cols_to_encode],
                drop_first=drop_first,
                dummy_na=dummy_na,
                prefix=cols_to_encode,
                prefix_sep=prefix_sep,
            )
            rest = feats.drop(columns=cols_to_encode)
            encoded = pd.concat([rest, feats[cols_to_encode], dummies], axis=1)
        else:
            encoded = pd.get_dummies(
                feats,
                columns=cols_to_encode,
                drop_first=drop_first,
                dummy_na=dummy_na,
                prefix=cols_to_encode,
                prefix_sep=prefix_sep,
            )

    if bool_to_number:
        bool_cols = [c for c in encoded.columns if pd.api.types.is_bool_dtype(encoded[c])]
        if bool_cols:
            encoded[bool_cols] = encoded[bool_cols].astype("int64")
    # Reattach id
    if ids is not None:
        encoded.insert(0, "id", ids.values)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    encoded.to_csv(output_csv, index=False)

    print(f"Loaded:  {input_csv} shape={df.shape}")
    print(f"bool_to_number={bool_to_number}")
    print(f"Encoded columns: {cols_to_encode}")
    print(f"Saved:   {output_csv} shape={encoded.shape}")

    return output_csv


one_hot_encode_csv("data/uci_dataset17.csv", "data/uci_dataset17_encoded.csv", columns="", drop_first=False,
                   dummy_na=True)
