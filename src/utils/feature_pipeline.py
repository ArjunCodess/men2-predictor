import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

BASE_CONTINUOUS_FEATURES = ['age', 'ret_risk_level', 'calcitonin_level_numeric']
BINARY_FEATURES = [
    'gender',
    'calcitonin_elevated',
    'thyroid_nodules_present',
    'multiple_nodules',
    'family_history_mtc',
    'pheochromocytoma',
    'hyperparathyroidism'
]
ENGINEERED_FEATURES = [
    'age_squared',
    'calcitonin_age_interaction',
    'risk_calcitonin_interaction',
    'risk_age_interaction',
    'nodule_severity'
]
CATEGORICAL_FEATURES = ['ret_variant', 'age_group']

PIPELINE_NUMERIC_COLUMNS = BASE_CONTINUOUS_FEATURES + BINARY_FEATURES + ENGINEERED_FEATURES
PIPELINE_FEATURE_COLUMNS = PIPELINE_NUMERIC_COLUMNS + CATEGORICAL_FEATURES


def _ensure_numeric_series(series: pd.Series, as_int: bool = False) -> pd.Series:
    """convert series to numeric safely."""
    numeric = pd.to_numeric(series, errors='coerce')
    if as_int:
        return numeric.fillna(0).astype(int)
    return numeric.fillna(0.0)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """add deterministic interaction features used by the shared encoder."""
    df = df.copy()
    df['age_squared'] = df['age'] ** 2
    df['calcitonin_age_interaction'] = df['calcitonin_level_numeric'] * df['age']
    df['risk_calcitonin_interaction'] = df['ret_risk_level'] * df['calcitonin_level_numeric']
    df['risk_age_interaction'] = df['ret_risk_level'] * df['age']
    df['nodule_severity'] = df['thyroid_nodules_present'] * df['multiple_nodules']
    return df


def prepare_patient_level_frame(df: pd.DataFrame) -> pd.DataFrame:
    """standardize feature columns and add engineered metrics."""
    df = df.copy()

    for col in BASE_CONTINUOUS_FEATURES:
        df[col] = _ensure_numeric_series(df.get(col, 0.0))

    gender_series = df.get('gender', 0)
    if hasattr(gender_series, 'dtype') and gender_series.dtype == object:
        gender_series = gender_series.str.strip().str.lower().map({'female': 0, 'male': 1})
    df['gender'] = _ensure_numeric_series(gender_series, as_int=True)

    for col in BINARY_FEATURES:
        if col == 'gender':
            continue
        df[col] = _ensure_numeric_series(df.get(col, 0), as_int=True)

    df['ret_variant'] = df.get('ret_variant', 'unknown').fillna('unknown').astype(str)
    df['age_group'] = df.get('age_group', 'unknown').fillna('unknown').astype(str)

    df = add_engineered_features(df)

    for col in ENGINEERED_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    missing_cols = set(PIPELINE_FEATURE_COLUMNS) - set(df.columns)
    for col in missing_cols:
        if col in CATEGORICAL_FEATURES:
            df[col] = 'unknown'
        else:
            df[col] = 0.0

    return df


def get_pipeline_ready_frame(df: pd.DataFrame) -> pd.DataFrame:
    """return ordered dataframe ready for the shared encoder."""
    prepared = prepare_patient_level_frame(df)
    return prepared[PIPELINE_FEATURE_COLUMNS]


def _create_one_hot_encoder() -> OneHotEncoder:
    """create one-hot encoder compatible with multiple sklearn versions."""
    init_params = OneHotEncoder.__init__.__code__.co_varnames
    kwargs = {'handle_unknown': 'ignore'}
    if 'sparse_output' in init_params:
        kwargs['sparse_output'] = False
    else:
        kwargs['sparse'] = False
    return OneHotEncoder(**kwargs)


def build_patient_feature_pipeline(pca_variance: float = 0.95) -> Pipeline:
    """create the shared encoder transformer for patient-level features."""
    numeric_cols = PIPELINE_NUMERIC_COLUMNS
    categorical_cols = CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('variance', VarianceThreshold(threshold=0.0))
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', _create_one_hot_encoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    pca = PCA(n_components=pca_variance, svd_solver='full', random_state=42)

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', pca)
    ])


__all__ = [
    'PIPELINE_FEATURE_COLUMNS',
    'build_patient_feature_pipeline',
    'get_pipeline_ready_frame',
    'prepare_patient_level_frame',
    'add_engineered_features'
]
