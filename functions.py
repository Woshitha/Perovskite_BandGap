import os
import re
import gc
import time
import json
import shutil
import logging
import warnings
import collections
from datetime import datetime
import numpy as np
import pandas as pd
import tqdm
import pkg_resources
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from scipy.optimize import minimize
import dcor
import shap
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    cross_val_predict,
    cross_validate)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score)
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    Ridge)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor)
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
from mapie.regression import ConformalizedQuantileRegressor as MapieRegressor
warnings.filterwarnings("ignore")


#######################################################################################################################
#                                        Feature Engineering
#######################################################################################################################

class CompositionError(Exception):
    """Exception class for composition errors"""
    pass


def get_sym_dict(f, factor):
    sym_dict = collections.defaultdict(float)
    # compile regex for speedup
    regex = r"([A-Z][a-z]*)\s*([-*\.\d]*)"
    r = re.compile(regex)
    for m in re.finditer(r, f):
        el = m.group(1)
        amt = 1
        if m.group(2).strip() != "":
            amt = float(m.group(2))
        sym_dict[el] += amt * factor
        f = f.replace(m.group(), "", 1)
    if f.strip():
        raise CompositionError(f'{f} is an invalid formula!')
    return sym_dict


def parse_formula(formula):
    # for Metallofullerene like "Y3N@C80"
    formula = formula.replace('@', '')
    formula = formula.replace('[', '(')
    formula = formula.replace(']', ')')
    # compile regex for speedup
    regex = r"\(([^\(\)]+)\)\s*([\.\d]*)"
    r = re.compile(regex)
    m = re.search(r, formula)
    if m:
        factor = 1
        if m.group(2) != "":
            factor = float(m.group(2))
        unit_sym_dict = get_sym_dict(m.group(1), factor)
        expanded_sym = "".join(["{}{}".format(el, amt)
                                for el, amt in unit_sym_dict.items()])
        expanded_formula = formula.replace(m.group(), expanded_sym)
        return parse_formula(expanded_formula)
    sym_dict = get_sym_dict(formula, 1)
    return sym_dict


def _fractional_composition(formula):
    elmap = parse_formula(formula)
    elamt = {}
    natoms = 0
    for k, v in elmap.items():
        if abs(v) >= 1e-6:
            elamt[k] = v
            natoms += abs(v)
    comp_frac = {key: elamt[key] / natoms for key in elamt}
    return comp_frac


def _fractional_composition_L(formula):
    comp_frac = _fractional_composition(formula)
    atoms = list(comp_frac.keys())
    counts = list(comp_frac.values())
    return atoms, counts


def _element_composition(formula):
    elmap = parse_formula(formula)
    elamt = {}
    natoms = 0
    for k, v in elmap.items():
        if abs(v) >= 1e-6:
            elamt[k] = v
            natoms += abs(v)
    return elamt


def _element_composition_L(formula):
    comp_frac = _element_composition(formula)
    atoms = list(comp_frac.keys())
    counts = list(comp_frac.values())
    return atoms, counts


def _assign_features(matrices, elem_info, formulae, sum_feat=False):
    formula_mat, count_mat, frac_mat, elem_mat, target_mat = matrices
    elem_symbols, elem_index, elem_missing = elem_info

    if sum_feat:
        sum_feats = []
    avg_feats = []
    range_feats = []
    # var_feats = []
    dev_feats = []
    max_feats = []
    min_feats = []
    mode_feats = []
    targets = []
    formulas = []
    skipped_formula = []

    for h in tqdm.tqdm(range(len(formulae)), desc='Assigning Features...'):
        elem_list = formula_mat[h]
        target = target_mat[h]
        formula = formulae[h]
        comp_mat = np.zeros(shape=(len(elem_list), elem_mat.shape[-1]))
        skipped = False

        for i, elem in enumerate(elem_list):
            if elem in elem_missing:
                skipped = True
            else:
                row = elem_index[elem_symbols.index(elem)]
                comp_mat[i, :] = elem_mat[row]

        if skipped:
            skipped_formula.append(formula)

        range_feats.append(np.ptp(comp_mat, axis=0))
        # var_feats.append(comp_mat.var(axis=0))
        max_feats.append(comp_mat.max(axis=0))
        min_feats.append(comp_mat.min(axis=0))

        comp_frac_mat = comp_mat.T * frac_mat[h]
        comp_frac_mat = comp_frac_mat.T
        avg_feats.append(comp_frac_mat.sum(axis=0))

        dev = np.abs(comp_mat - comp_frac_mat.sum(axis=0))
        dev = dev.T * frac_mat[h]
        dev = dev.T.sum(axis=0)
        dev_feats.append(dev)

        prominant = np.isclose(frac_mat[h], max(frac_mat[h]))
        mode = comp_mat[prominant].min(axis=0)
        mode_feats.append(mode)

        comp_sum_mat = comp_mat.T * count_mat[h]
        comp_sum_mat = comp_sum_mat.T
        if sum_feat:
            sum_feats.append(comp_sum_mat.sum(axis=0))

        targets.append(target)
        formulas.append(formula)

    if len(skipped_formula) > 0:
        print('\nNOTE: Your data contains formula with exotic elements.',
              'These were skipped.')
    if sum_feat:
        conc_list = [sum_feats, avg_feats, dev_feats,
                     range_feats, max_feats, min_feats, mode_feats]
        feats = np.concatenate(conc_list, axis=1)
    else:
        conc_list = [avg_feats, dev_feats,
                     range_feats, max_feats, min_feats, mode_feats]
        feats = np.concatenate(conc_list, axis=1)

    return feats, targets, formulas, skipped_formula


def generate_features(df, elem_prop='oliynyk',
                      drop_duplicates=False,
                      extend_features=False,
                      sum_feat=False,
                      mini=False,
                      req_features=None):
    
    if drop_duplicates:
        if df['formula'].value_counts()[0] > 1:
            df.drop_duplicates('formula', inplace=True)
            print('Duplicate formula(e) removed using default pandas function')

    all_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                   'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
                   'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                   'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                   'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                   'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                   'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                   'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                   'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                   'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                   'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                   'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']


    elem_props = pd.read_csv(f'element_properties/{elem_prop}.csv')

    elem_props.index = elem_props['element'].values
    elem_props.drop(['element'], inplace=True, axis=1)

    elem_symbols = elem_props.index.tolist()
    elem_index = np.arange(0, elem_props.shape[0], 1)
    elem_missing = list(set(all_symbols) - set(elem_symbols))

    elem_props_columns = elem_props.columns.values

    column_names = np.concatenate(['avg_' + elem_props_columns,
                                   'dev_' + elem_props_columns,
                                   'range_' + elem_props_columns,
                                   'max_' + elem_props_columns,
                                   'min_' + elem_props_columns,
                                   'mode_' + elem_props_columns])
    if sum_feat:
        column_names = np.concatenate(['sum_' + elem_props_columns,
                                       column_names])

    # make empty list where we will store the property value
    targets = []
    # store formula
    formulae = []
    # add the values to the list using a for loop

    elem_mat = elem_props.values

    formula_mat = []
    count_mat = []
    frac_mat = []
    target_mat = []

    if extend_features:
        features = df.columns.values.tolist()
        features.remove('target')
        extra_features = df[features]

    for index in tqdm.tqdm(df.index.values, desc='Processing Input Data'):
        formula, target = df.loc[index, 'formula'], df.loc[index, 'target']
        if 'x' in formula:
            continue
        l1, l2 = _element_composition_L(formula)
        formula_mat.append(l1)
        count_mat.append(l2)
        _, l3 = _fractional_composition_L(formula)
        frac_mat.append(l3)
        target_mat.append(target)
        formulae.append(formula)

    print('\tfeaturizing compositions...'.title())

    matrices = [formula_mat, count_mat, frac_mat, elem_mat, target_mat]
    elem_info = [elem_symbols, elem_index, elem_missing]
    feats, targets, formulae, skipped = _assign_features(matrices,
                                                         elem_info,
                                                         formulae,
                                                         sum_feat=sum_feat)

    print('\tcreating pandas objects...'.title())

    # split feature vectors and targets as X and y
    X = pd.DataFrame(feats, columns=column_names, index=formulae)
    y = pd.Series(targets, index=formulae, name='target')
    formulae = pd.Series(formulae, index=formulae, name='formula')
    if extend_features:
        extended = pd.DataFrame(extra_features, columns=features)
        extended = extended.set_index('formula', drop=True)
        X = pd.concat([X, extended], axis=1)

    # reset dataframe indices
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    formulae.reset_index(drop=True, inplace=True)

    # drop elements that aren't included in the elmenetal properties list.
    # These will be returned as feature rows completely full of NaN values.
    X.dropna(inplace=True, how='all')
    y = y.iloc[X.index]
    formulae = formulae.iloc[X.index]

    # get the column names
    cols = X.columns.values
    # find the median value of each column
    median_values = X[cols].median()
    # fill the missing values in each column with the column's median value
    X[cols] = X[cols].fillna(median_values)

    # Only return the avg/sum of element properties.

    if mini:
        np.random.seed(42)
        booleans = np.random.rand(X.shape[-1]) <= 64/X.shape[-1]
        X = X.iloc[:, booleans]

    # Filter features based on req_features.csv if provided
    if req_features is not None:
        required_features_df = pd.read_csv(req_features, header=None)
        required_features = required_features_df[0].tolist()
        
        # Find common features between X and required_features
        common_features = [feat for feat in required_features if feat in X.columns]
        
        # Filter X to keep only required features
        X = X[common_features]
        
        # Print info about filtered features
        missing_features = set(required_features) - set(X.columns)
        if missing_features:
            print(f'\nNote: {len(missing_features)} required features were not found in generated features.')

    return X, y, formulae, skipped


def feature_selection_pipeline(df, params=None, save = (False, "feature_selection.csv")):
    # Default parameters
    default_params = {
        "var_threshold": 1e-5,
        "duplicate_corr_threshold": 0.95,
        "dcor_threshold1": 0.2,
        "mi_threshold1": 0.1,
        "cutoff_Mimp": 0,
        "dcor_and_mi": False, #True = dcor And mi, False = dcor Or mi
        "random_state": 42,
        "lightgbm_params": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "random_state": 42,
            "verbose": -1
        }
    }
    
    # Merge with user parameters
    config = {**default_params, **(params or {})}
    
    print("\n\nStarting feature selection pipeline...")
    print(f"\nInitial data shape: {df.shape}")
    
    # Extract components
    target = df.iloc[:, 1]
    features = df.iloc[:, 2:]
    
    print(f"Initial features: {features.shape[1]}")
    
    # === STEP 1: Quick Pre-Filtering ===
    print("\n--- Step 1: Quick Pre-Filtering ---")
    
    # Remove near-zero variance features
    initial_count = features.shape[1]
    feature_variances = features.var()
    keep_vars = feature_variances[feature_variances > config["var_threshold"]].index
    features = features[keep_vars]
    print(f"Removed {initial_count - features.shape[1]} low-variance features")
    
    # Remove highly correlated duplicates
    if features.shape[1] > 1:
        initial_count = features.shape[1]
        corr_matrix = features.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_tri.columns 
                  if any(upper_tri[col] > config["duplicate_corr_threshold"])]
        features = features.drop(columns=to_drop)
        print(f"Removed {len(to_drop)} highly correlated features")
    
    del corr_matrix, feature_variances
    gc.collect()

    print(f"Features after pre-filtering: {features.shape[1]}")
    
    # === STEP 2: Statistical Filtering ===
    print("\n--- Step 2: Statistical Filtering ---")
    
    print("Computing distance correlations...")
    dcor_scores = []
    for col in features.columns:
        try:
            dcor_val = dcor.distance_correlation(features[col], target)
            dcor_scores.append(dcor_val)
        except:
            dcor_scores.append(0.0)
    
    print("Computing mutual information...")
    try:
        mi_scores = mutual_info_regression(features, target, 
                                         random_state=config["random_state"])
    except:
        mi_scores = [0.0] * len(features.columns)
    
    # Create statistics dataframe
    stat_df = pd.DataFrame({
        "feature": features.columns,
        "dcor": dcor_scores,
        "mi": mi_scores
    })
    
    # Determine which features pass step 2 
    if config["dcor_and_mi"]:
        # ANDR condition: pass both dcor and mi threshold
        stat_df["pass_step2"] = ((stat_df["dcor"] > config["dcor_threshold1"]) & (stat_df["mi"] > config["mi_threshold1"]))
    else:
        # OR condition: pass dcor OR mi threshold
        stat_df["pass_step2"] = ((stat_df["dcor"] > config["dcor_threshold1"]) | (stat_df["mi"] > config["mi_threshold1"]))
    
    # Get features that passed step 2
    passed_features = stat_df[stat_df["pass_step2"]]["feature"].tolist()
    
    print(f"Features passing dCor threshold ({config['dcor_threshold1']}): {len(stat_df[stat_df['dcor'] > config['dcor_threshold1']])}")
    print(f"Features passing MI threshold ({config['mi_threshold1']}): {len(stat_df[stat_df['mi'] > config['mi_threshold1']])}")
    print(f"Features selected for model training: {len(passed_features)}")
    
    # === STEP 3: Model-Based Ranking ===
    
    print("\n--- Step 3: Model-Based Ranking ---")
    
    # Initialize model importance as NaN for all features
    stat_df["model_importance"] = np.nan
    
    if len(passed_features) > 10:
        try:
            # Use only features that passed statistical filtering for model training
            features_for_model = features[passed_features]
            
            # Clean feature names for LightGBM
            clean_names = {col: re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in features_for_model.columns}
            features_clean = features_for_model.rename(columns=clean_names)
            
            print(f"Training LightGBM model on {len(passed_features)} features...")
            model = LGBMRegressor(**config["lightgbm_params"])
            model.fit(features_clean, target)
            
            # Get feature importances only for features used in model training
            model_importances = pd.Series(model.feature_importances_, 
                                        index=features_for_model.columns)
            
            # Update model_importance column for features that were used in training
            for feature in passed_features:
                stat_df.loc[stat_df["feature"] == feature, "model_importance"] = model_importances[feature]
            
            print(f"Model training completed successfully")
                
        except Exception as e:
            print(f"Warning: Model-based selection failed: {e}")
            print("Model importance values will remain as NaN")
    else:
        print(f"Skipping model training (only {len(passed_features)} features passed statistical filtering, minimum required: 10)")
        print("Model importance values will remain as NaN")
    
    # Reorder columns as requested: [feature, dcor, MI, Model_importance, pass_step2]
    feature_importance = stat_df[["feature", "dcor", "mi", "model_importance", "pass_step2"]]

    # Print summary
    print(f"\n\n=== FEATURE SELECTION SUMMARY ===")
    print(f"Initial features: {df.shape[1] - 2}")
    print(f"Features after step 1 (pre-filtering): {features.shape[1]}")
    print(f"Features after step 2 (statistical filtering): {len(passed_features)}")
    print(f"Features after step 3 (model importance > {config['cutoff_Mimp']}): {(stat_df['model_importance'] > config['cutoff_Mimp']).sum()}")


    # Save selected features as csv
    df_filtered = feature_importance[feature_importance['pass_step2'] == True]
    feature_only = df_filtered[['feature']]
    feature_only.to_csv('selected_features.csv', index=False, header=False)

    # Filter selected features
    first_two_cols = df.columns[:2].tolist()
    feature_cols = df.columns[2:]
    feature_only_list = feature_only.iloc[:, 0].tolist()
    selected_features = [f for f in feature_cols if f in feature_only_list]
    final_columns = first_two_cols + selected_features
    filtered_df = df[final_columns]
    filtered_df.to_csv("Filtered_Features.csv", index=False)

    # Save feature selection results
    if save[0] == True:
        feature_importance.to_csv(f"{save[1]}", index=False)
        print(f"Updated file saved to: {save[1]}")
    
    return feature_importance


#######################################################################################################################
#                                        Classification model training
#######################################################################################################################

RANDOM_STATE = 42

# Setup logging
def setup_logging(output_dir='classification_outputs'):
    """Setup logging configuration to log to both console and file."""
    os.makedirs(output_dir, exist_ok=True)
    log_filename = os.path.join(output_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_filename}")
    return log_filename


def save_metadata(metadata, output_dir='classification_outputs'):
    """Save model metadata to JSON file."""
    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Metadata saved to: {metadata_path}")


def train_classification_model(df, bandgap_threshold: float, test_size: float = 0.2, output_dir='classification_outputs'):
    # Setup logging
    setup_logging(output_dir)
    
    logging.info("="*80)
    logging.info("CLASSIFICATION MODEL TRAINING PIPELINE")
    logging.info("="*80)
    logging.info(f"Band gap threshold: {bandgap_threshold}")
    logging.info(f"Test size: {test_size}")
    logging.info(f"Random state: {RANDOM_STATE}")
    logging.info("-" * 80)
    logging.info(f"Dataset shape: {df.shape}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Drop structure column (column 0) and extract features
    X = df.iloc[:, 2:].values  # Features from column 2 onward
    bandgaps = df.iloc[:, 1].values  # Band gap values from column 1
    feature_names = df.columns[2:].tolist()
    
    del df
    gc.collect()

    # Create binary labels based on threshold
    y = (bandgaps >= bandgap_threshold).astype(int)
    
    logging.info(f"Features shape: {X.shape}")
    logging.info(f"Number of features: {X.shape[1]}")
    logging.info(f"Class distribution: Class 0={np.sum(y==0)}, Class 1={np.sum(y==1)}")
    logging.info(f"Class balance ratio: {np.sum(y==1)/len(y):.2%}")
    
    # ============================================================
    # STEP 1: ROBUST DATA SPLITTING WITH STRATIFICATION
    # ============================================================
    logging.info("-" * 80)
    logging.info("STEP 1: Performing stratified train-test split...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    
    logging.info(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/len(X):.1%})")
    logging.info(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X):.1%})")
    logging.info(f"Training set class distribution: Class 0={np.sum(y_train==0)}, Class 1={np.sum(y_train==1)}")
    logging.info(f"Test set class distribution: Class 0={np.sum(y_test==0)}, Class 1={np.sum(y_test==1)}")
    
    # ============================================================
    # STEP 2: FEATURE STANDARDIZATION AND SCALER SAVING
    # ============================================================
    logging.info("-" * 80)
    logging.info("STEP 2: Standardizing features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for deployment
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved to: {scaler_path}")
    logging.info(f"Feature mean: {scaler.mean_[:5]}... (showing first 5)")
    logging.info(f"Feature std: {scaler.scale_[:5]}... (showing first 5)")
    
    # ============================================================
    # STEP 3: HANDLE CLASS IMBALANCE WITH SMOTE
    # ============================================================
    logging.info("-" * 80)
    logging.info("STEP 3: Applying SMOTE for class balancing...")
    
    X_train_balanced, y_train_balanced = clz_balance(X_train_scaled, y_train)
    
    # ============================================================
    # STEP 4: MODEL DEFINITION AND HYPERPARAMETER GRIDS
    # ============================================================
    logging.info("-" * 80)
    logging.info("STEP 4: Defining models and hyperparameter grids...")
    
    models = {
        'Logistic_Regression': {
            'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        },
        'SVM': {
            'model': SVC(random_state=RANDOM_STATE, probability=True),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'Decision_Tree': {
            'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'params': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Random_Forest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Gradient_Boosting': {
            'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'Ridge_Classifier': {
            'model': RidgeClassifier(random_state=RANDOM_STATE),
            'params': {
                'alpha': [0.01, 0.1, 1, 10, 100],
                'solver': ['auto', 'svd', 'cholesky']
            }
        },
        'Extra_Trees': {
            'model': ExtraTreesClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    }
    
    logging.info(f"Total models to train: {len(models)}")
    
    # ============================================================
    # STEP 5: MODEL TRAINING WITH CROSS-VALIDATION
    # ============================================================
    logging.info("-" * 80)
    logging.info("STEP 5: Training models with GridSearchCV and cross-validation...")
    
    results = []
    trained_models = {}
    
    for model_name, model_config in models.items():
        logging.info(f"\n{'='*60}")
        logging.info(f"Training: {model_name}")
        logging.info(f"{'='*60}")
        
        try:
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model_config['model'], 
                model_config['params'], 
                cv=5, 
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            # Fit model
            start_time = time.time()
            grid_search.fit(X_train_balanced, y_train_balanced)
            training_time = time.time() - start_time
            
            best_model = grid_search.best_estimator_
            logging.info(f"Best parameters: {grid_search.best_params_}")
            logging.info(f"Training time: {training_time:.2f}s")
            
            # ============================================================
            # CROSS-VALIDATION SCORING
            # ============================================================
            logging.info("Performing 5-fold cross-validation...")
            cv_scores = cross_validate(
                best_model, 
                X_train_balanced, 
                y_train_balanced,
                cv=5,
                scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                n_jobs=-1
            )
            
            cv_accuracy = cv_scores['test_accuracy'].mean()
            cv_f1 = cv_scores['test_f1'].mean()
            cv_precision = cv_scores['test_precision'].mean()
            cv_recall = cv_scores['test_recall'].mean()
            cv_roc_auc = cv_scores['test_roc_auc'].mean()
            
            logging.info(f"Cross-validation results (5-fold):")
            logging.info(f"  - Accuracy: {cv_accuracy:.4f} ± {cv_scores['test_accuracy'].std():.4f}")
            logging.info(f"  - F1 Score: {cv_f1:.4f} ± {cv_scores['test_f1'].std():.4f}")
            logging.info(f"  - Precision: {cv_precision:.4f} ± {cv_scores['test_precision'].std():.4f}")
            logging.info(f"  - Recall: {cv_recall:.4f} ± {cv_scores['test_recall'].std():.4f}")
            logging.info(f"  - ROC-AUC: {cv_roc_auc:.4f} ± {cv_scores['test_roc_auc'].std():.4f}")
            
            # ============================================================
            # TEST SET EVALUATION
            # ============================================================
            logging.info("Evaluating on test set...")
            y_pred = best_model.predict(X_test_scaled)
            
            # Get prediction probabilities for ROC-AUC
            if hasattr(best_model, 'predict_proba'):
                y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
            elif hasattr(best_model, 'decision_function'):
                y_proba = best_model.decision_function(X_test_scaled)
            else:
                y_proba = y_pred
            
            # Calculate test metrics
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred)
            test_recall = recall_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred)
            test_roc_auc = roc_auc_score(y_test, y_proba)
            
            logging.info(f"Test set results:")
            logging.info(f"  - Accuracy: {test_accuracy:.4f}")
            logging.info(f"  - F1 Score: {test_f1:.4f}")
            logging.info(f"  - Precision: {test_precision:.4f}")
            logging.info(f"  - Recall: {test_recall:.4f}")
            logging.info(f"  - ROC-AUC: {test_roc_auc:.4f}")
            
            # Store results
            results.append({
                'Model': model_name,
                'CV_Accuracy': cv_accuracy,
                'CV_F1': cv_f1,
                'CV_Precision': cv_precision,
                'CV_Recall': cv_recall,
                'CV_ROC_AUC': cv_roc_auc,
                'Test_Accuracy': test_accuracy,
                'Test_Precision': test_precision,
                'Test_Recall': test_recall,
                'Test_F1_Score': test_f1,
                'Test_ROC_AUC': test_roc_auc,
                'Training_Time': training_time,
                'Best_Params': str(grid_search.best_params_)
            })
            
            # Store model data
            trained_models[model_name] = {
                'model': best_model,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'roc_auc': test_roc_auc,
                'y_test': y_test,
                'X_test': X_test_scaled,
                'cv_scores': cv_scores
            }
            
        except Exception as e:
            logging.error(f"Error training {model_name}: {str(e)}")
            continue
    
    # ============================================================
    # STEP 6: MODEL SELECTION AND BEST MODEL SAVING
    # ============================================================
    logging.info("\n" + "="*80)
    logging.info("STEP 6: Selecting and saving best model...")
    logging.info("="*80)
    
    # Save evaluation metrics to CSV
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test_ROC_AUC', ascending=False)
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.csv')
    results_df.to_csv(metrics_path, index=False)
    logging.info(f"Evaluation metrics saved to: {metrics_path}")
    
    # Find best model
    best_model_name = results_df.iloc[0]['Model']
    best_model_obj = trained_models[best_model_name]['model']
    best_roc_auc = trained_models[best_model_name]['roc_auc']
    best_model_results = results_df.iloc[0]
    
    logging.info(f"\n{'*'*80}")
    logging.info(f"BEST MODEL: {best_model_name}")
    logging.info(f"{'*'*80}")
    logging.info(f"Test ROC-AUC: {best_roc_auc:.4f}")
    logging.info(f"Test Accuracy: {best_model_results['Test_Accuracy']:.4f}")
    logging.info(f"Test F1 Score: {best_model_results['Test_F1_Score']:.4f}")
    logging.info(f"Test Precision: {best_model_results['Test_Precision']:.4f}")
    logging.info(f"Test Recall: {best_model_results['Test_Recall']:.4f}")
    
    # Save best model
    best_model_path = os.path.join(output_dir, 'best_model.pkl')
    joblib.dump({
        'model': best_model_obj,
        'scaler': scaler,
        'threshold': bandgap_threshold,
        'feature_names': feature_names,
        'model_name': best_model_name
    }, best_model_path)
    logging.info(f"Best model saved to: {best_model_path}")
    
    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'model_type': str(type(best_model_obj).__name__),
        'parameters': str(best_model_obj.get_params()),
        'bandgap_threshold': bandgap_threshold,
        'test_roc_auc': float(best_roc_auc),
        'test_accuracy': float(best_model_results['Test_Accuracy']),
        'test_f1_score': float(best_model_results['Test_F1_Score']),
        'test_precision': float(best_model_results['Test_Precision']),
        'test_recall': float(best_model_results['Test_Recall']),
        'cv_roc_auc_mean': float(best_model_results['CV_ROC_AUC']),
        'cv_accuracy_mean': float(best_model_results['CV_Accuracy']),
        'training_time_seconds': float(best_model_results['Training_Time']),
        'n_features': X.shape[1],
        'n_training_samples': X_train_balanced.shape[0],
        'n_test_samples': X_test.shape[0],
        'random_state': RANDOM_STATE,
        'timestamp': datetime.now().isoformat()
    }
    save_metadata(metadata, output_dir)
    
    # ============================================================
    # STEP 7: COMPREHENSIVE REPORTING
    # ============================================================
    logging.info("\n" + "="*80)
    logging.info("STEP 7: Generating comprehensive reports...")
    logging.info("="*80)
    
    # Print detailed classification report for best model
    logging.info(f"\nDetailed Classification Report for {best_model_name}:")
    logging.info("\n" + classification_report(
        trained_models[best_model_name]['y_test'], 
        trained_models[best_model_name]['y_pred'],
        target_names=['Class 0', 'Class 1']
    ))
    
    # Print confusion matrix
    cm = confusion_matrix(
        trained_models[best_model_name]['y_test'], 
        trained_models[best_model_name]['y_pred']
    )
    logging.info(f"\nConfusion Matrix for {best_model_name}:")
    logging.info(f"\n{cm}")
    logging.info(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    logging.info(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    # Print summary table
    logging.info("\n" + "="*80)
    logging.info("MODEL PERFORMANCE SUMMARY (sorted by Test ROC-AUC)")
    logging.info("="*80)
    summary_cols = ['Model', 'Test_ROC_AUC', 'Test_Accuracy', 'Test_F1_Score', 
                    'Test_Precision', 'Test_Recall', 'CV_ROC_AUC']
    logging.info("\n" + results_df[summary_cols].to_string(index=False))
    
    # ============================================================
    # STEP 8: GENERATE VISUALIZATIONS
    # ============================================================
    logging.info("\n" + "="*80)
    logging.info("STEP 8: Generating visualizations...")
    logging.info("="*80)
    
    generate_combined_plots(trained_models, output_dir)
    
    
    # ============================================================
    # STEP 9: FEATURE IMPORTANCE AND SHAP ANALYSIS
    # ============================================================
    logging.info("\n" + "="*80)
    logging.info("STEP 9: Generating feature importance and SHAP analysis...")
    logging.info("="*80)
    
    
    generate_feature_importance_and_shap(
        best_model_obj, 
        best_model_name,
        X_train_balanced, 
        X_test_scaled,
        feature_names,
        output_dir
    )
    
    logging.info("\n" + "="*80)
    logging.info("TRAINING COMPLETED SUCCESSFULLY!")
    logging.info("="*80)
    logging.info(f"All outputs saved to: {output_dir}/")
    
    return {
        'best_model_name': best_model_name,
        'best_roc_auc': best_roc_auc,
        'results_df': results_df,
        #'trained_models': trained_models,
        'metadata': metadata
    }


def clz_balance(X_train, y_train):
    """Apply SMOTE for class balancing."""
    logging.info("Applying SMOTE for class balancing...")
    logging.info(f"Before SMOTE: Class 0={np.sum(y_train==0)}, Class 1={np.sum(y_train==1)}")
    
    smote = SMOTE(sampling_strategy='auto', random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    logging.info(f"After SMOTE: Class 0={np.sum(y_res==0)}, Class 1={np.sum(y_res==1)}")
    return X_res, y_res


def generate_combined_plots(trained_models, output_dir='classification_outputs'):
    """
    Generate combined plots for all models: ROC curves, confusion matrices, and uncertainty plots.
    """
    logging.info("Generating combined visualization plots...")
    plt.style.use('default')
    model_names = list(trained_models.keys())
    n_models = len(model_names)
    
    # 1. Combined ROC Curves
    logging.info("Creating combined ROC curves...")
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    for i, (model_name, model_data) in enumerate(trained_models.items()):
        model = model_data['model']
        y_test = model_data['y_test']
        y_proba = model_data['y_proba']
        
        if hasattr(model, 'predict_proba') or hasattr(model, 'decision_function'):
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = model_data['roc_auc']
        else:
            fpr, tpr, _ = roc_curve(y_test, model_data['y_pred'])
            auc = model_data['roc_auc']
        
        plt.plot(fpr, tpr, linewidth=2, color=colors[i], 
                label=f'{model_name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison - All Models', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, 'combined_roc_curves.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"ROC curves saved to: {roc_path}")
    
    # 2. Combined Confusion Matrices
    logging.info("Creating combined confusion matrices...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (model_name, model_data) in enumerate(trained_models.items()):
        cm = confusion_matrix(model_data['y_test'], model_data['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'], ax=axes[i])
        axes[i].set_title(f'{model_name}', fontweight='bold')
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'combined_confusion_matrices.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Confusion matrices saved to: {cm_path}")
    
    # 3. Combined Uncertainty Plots
    logging.info("Creating combined uncertainty plots...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, (model_name, model_data) in enumerate(trained_models.items()):
        model = model_data['model']
        X_test = model_data['X_test']
        y_test = model_data['y_test']
        
        if hasattr(model, 'predict_proba'):
            proba_class_1 = model.predict_proba(X_test)[:, 1]
            axes[i].hist(proba_class_1[y_test == 0], bins=15, alpha=0.6, 
                        label='True Class 0', color='red', density=True)
            axes[i].hist(proba_class_1[y_test == 1], bins=15, alpha=0.6, 
                        label='True Class 1', color='blue', density=True)
            axes[i].set_xlabel('Predicted Probability (Class 1)')
            
        elif hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(X_test)
            axes[i].hist(decision_scores[y_test == 0], bins=15, alpha=0.6, 
                        label='True Class 0', color='red', density=True)
            axes[i].hist(decision_scores[y_test == 1], bins=15, alpha=0.6, 
                        label='True Class 1', color='blue', density=True)
            axes[i].set_xlabel('Decision Function Score')
            
        else:
            y_pred = model_data['y_pred']
            unique, counts = np.unique(y_pred, return_counts=True)
            axes[i].bar(unique, counts/len(y_pred), alpha=0.7, color=['red', 'blue'])
            axes[i].set_xlabel('Predicted Class')
            axes[i].set_xticks([0, 1])
            axes[i].set_xticklabels(['Class 0', 'Class 1'])
        
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'{model_name}', fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Prediction Uncertainty - All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    uncertainty_path = os.path.join(output_dir, 'combined_uncertainty_plots.png')
    plt.savefig(uncertainty_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Uncertainty plots saved to: {uncertainty_path}")


def generate_feature_importance_and_shap(model, model_name, X_train, X_test, feature_names, output_dir='classification_outputs'):
    """
    Generate feature importance plots and SHAP analysis for the best model.
    """
    logging.info(f"Generating feature importance and SHAP analysis for {model_name}...")
    
    # Feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        logging.info("Creating feature importance plot...")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20 features
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Top 20 Feature Importances - {model_name}', fontsize=14, fontweight='bold')
        plt.barh(range(len(indices)), importances[indices], color='steelblue')
        plt.yticks(range(len(indices)), [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        importance_path = os.path.join(output_dir, f'feature_importance_{model_name}.png')
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Feature importance plot saved to: {importance_path}")
    
    # SHAP Analysis
    try:
        logging.info("Generating SHAP analysis...")
        
        # Sample data if too large (SHAP can be computationally expensive)
        max_samples = 1000
        if X_train.shape[0] > max_samples:
            sample_indices = np.random.choice(X_train.shape[0], max_samples, replace=False)
            X_train_sample = X_train[sample_indices]
            logging.info(f"Sampling {max_samples} training samples for SHAP analysis")
        else:
            X_train_sample = X_train
        
        if X_test.shape[0] > max_samples:
            test_indices = np.random.choice(X_test.shape[0], max_samples, replace=False)
            X_test_sample = X_test[test_indices]
            logging.info(f"Sampling {max_samples} test samples for SHAP analysis")
        else:
            X_test_sample = X_test
        
        # Choose appropriate SHAP explainer based on model type
        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, 
                             ExtraTreesClassifier, DecisionTreeClassifier, xgb.XGBClassifier)):
            logging.info("Using TreeExplainer for SHAP analysis...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, take positive class
            
        else:
            logging.info("Using KernelExplainer for SHAP analysis...")
            explainer = shap.KernelExplainer(model.predict_proba, X_train_sample)
            shap_values = explainer.shap_values(X_test_sample)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        
        # SHAP Summary Plot (Bar)
        logging.info("Creating SHAP summary bar plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_sample, 
                         feature_names=feature_names if len(feature_names) == X_test_sample.shape[1] else None,
                         plot_type="bar", show=False, max_display=20)
        plt.title(f'SHAP Feature Importance (Bar) - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        shap_bar_path = os.path.join(output_dir, f'shap_summary_bar_{model_name}.png')
        plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"SHAP bar plot saved to: {shap_bar_path}")
        
        # SHAP Summary Plot (Beeswarm)
        logging.info("Creating SHAP beeswarm plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_sample,
                         feature_names=feature_names if len(feature_names) == X_test_sample.shape[1] else None,
                         show=False, max_display=20)
        plt.title(f'SHAP Feature Impact (Beeswarm) - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        shap_beeswarm_path = os.path.join(output_dir, f'shap_summary_beeswarm_{model_name}.png')
        plt.savefig(shap_beeswarm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"SHAP beeswarm plot saved to: {shap_beeswarm_path}")
        
        logging.info("SHAP analysis completed successfully!")
        
    except Exception as e:
        logging.warning(f"SHAP analysis failed: {str(e)}")
        logging.warning("Continuing without SHAP plots...")



#######################################################################################################################
#                                        Classification - Model Inference functions
#######################################################################################################################

def load_classification_model(model_path):

    print(f"Loading model from: {model_path}")
    model_data = joblib.load(model_path)

    model = model_data['model']
    scaler = model_data['scaler']
    threshold = model_data.get('threshold', None)
    feature_names = model_data.get('feature_names', [])
    model_name = model_data.get('model_name', 'Unknown')

    print(f"Model loaded: {model_name} ({type(model).__name__})")
    print(f"Bandgap threshold: {threshold}")
    print(f"Number of features expected: {len(feature_names)}")
    print(f"Scaler type: {type(scaler).__name__}")

    return {
        'model': model,
        'scaler': scaler,
        'threshold': threshold,
        'feature_names': feature_names,
        'model_name': model_name
    }


def validate_and_align_features(featurized_df, required_features):

    # First column is structure/formula
    structure_col = featurized_df.iloc[:, 0]

    # All other columns are features
    available_feature_names = featurized_df.columns[1:].tolist()

    # Find intersection of required and available features
    features_to_use = [f for f in required_features if f in available_feature_names]
    missing_features = [f for f in required_features if f not in available_feature_names]

    if len(missing_features) > 0:
        print(f"Warning: {len(missing_features)} required features missing")
        print(f"Available: {len(features_to_use)}/{len(required_features)} features")

    if len(features_to_use) == 0:
        raise ValueError("No required features found in featurized data!")

    # Extract features in correct order
    X = featurized_df[features_to_use].values

    return structure_col, X, features_to_use


def get_predictions(model, X_scaled):

    predictions = model.predict(X_scaled)

    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_scaled)[:, 1]

    elif hasattr(model, 'decision_function'):
        decision_scores = model.decision_function(X_scaled)
        probabilities = (decision_scores - decision_scores.min()) / (
            decision_scores.max() - decision_scores.min()
        )

    else:
        probabilities = predictions.astype(float)

    return predictions, probabilities


def classify_structures(
    input_file="input.csv",
    model_path="model.joblib",
    output_file="classified_output.csv"
):

    print("=" * 80)
    print("CLASSIFICATION INFERENCE")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Model path: {model_path}")
    print(f"Output file: {output_file}")
    print("=" * 80)

    # Load model
    model_components = load_classification_model(model_path)
    model = model_components['model']
    scaler = model_components['scaler']
    required_features = model_components['feature_names']

    # Load input data
    print("\nLoading input structures...")
    df_input = pd.read_csv(input_file)

    if 'Structure' not in df_input.columns:
        raise ValueError("Input file must contain 'Structure' column")

    total_structures = len(df_input)
    print(f"Total structures: {total_structures}")

    # Validate and align features
    print("\nValidating features...")
    structure_col, X, features_used = validate_and_align_features(
        df_input,
        required_features
    )

    print(f"Feature matrix shape: {X.shape}")

    # Scale features
    print("\nScaling features...")
    X_scaled = scaler.transform(X)

    print(f"Scaled data range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")

    # Predictions
    print("\nMaking predictions...")
    predictions, probabilities = get_predictions(model, X_scaled)

    class_labels = np.where(predictions == 1, "High", "Low")

    print(f"Predictions - High: {np.sum(predictions == 1)}, Low: {np.sum(predictions == 0)}")

    # Create output dataframe
    output_df = pd.DataFrame({
        'Structure': structure_col,
        'predicted_class': class_labels,
        'probability_high': probabilities,
        'confidence': np.abs(probabilities - 0.5) * 2
    })

    # Save results
    output_df.to_csv(output_file, index=False)

    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total structures: {total_structures}")
    print(f"High bandgap predictions: {np.sum(predictions == 1)}")
    print(f"Low bandgap predictions: {np.sum(predictions == 0)}")
    print(f"Results saved to: {output_file}")
    print("=" * 80)

    return output_file


#######################################################################################################################
#                                        Regression model training
#######################################################################################################################


def train_individual_models(INPUT_CSV, BANDGAP_MIN, BANDGAP_MAX, TEST_SIZE=0.2, N_TRIALS=100, RANDOM_STATE=42):
    output_dir = f"regression_outputs_{BANDGAP_MIN}-{BANDGAP_MAX}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir+'/models', exist_ok=True)
    
    print("="*80)
    print("INDIVIDUAL MODEL TRAINING")
    print("="*80)
    
    # Load and filter data
    df = pd.read_csv(INPUT_CSV)
    bandgap_col = 'bandgap' if 'bandgap' in df.columns else 'BandGap'
    df_filtered = df[(df[bandgap_col] >= BANDGAP_MIN) & (df[bandgap_col] <= BANDGAP_MAX)].dropna()
    
    print(f"Filtered dataset: {df_filtered.shape}")
    
    # Plot bandgap distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df_filtered[bandgap_col], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Band Gap (eV)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of Band Gap Values\n(Range: {BANDGAP_MIN}-{BANDGAP_MAX} eV, N={len(df_filtered)})', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bandgap_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Prepare features and target
    X = df_filtered.iloc[:, 2:].values.astype(np.float32)
    y = df_filtered[bandgap_col].values.astype(np.float32)
    
    del df, df_filtered
    gc.collect()
    
    # Scale and split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    joblib.dump({
        'scaler': scaler,
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'bandgap_range': (BANDGAP_MIN, BANDGAP_MAX),
        # Also save the actual test set for verification
        'X_test': X_test,
        'y_test': y_test
    }, f'{output_dir}/models/scaler_and_split.joblib')
    
    # Train models (rest remains the same)
    model_configs = {
        'Random_Forest': optimize_random_forest,
        'Extra_Trees': optimize_extra_trees,
        'XGBoost': optimize_xgboost,
        'LightGBM': optimize_lightgbm,
        'CatBoost': optimize_catboost,
        'SVR': optimize_svr,
        'KNN': optimize_knn,
        'MLP': optimize_mlp,
        'Ridge': optimize_ridge,
        'Kernel_Ridge': optimize_kernel_ridge
    }
    
    results = []
    
    for model_name, optimize_func in model_configs.items():
        print(f"\n{'='*60}\n{model_name}\n{'='*60}")
        
        try:
            study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
            
            def print_progress(study, trial):
                if (trial.number + 1) % 10 == 0 or (trial.number + 1) == N_TRIALS:
                    print(f"Progress: {((trial.number + 1)/N_TRIALS)*100:.0f}% - Best MAE: {study.best_value:.4f}")
            
            study.optimize(lambda trial: optimize_func(trial, X_train, y_train), 
                          n_trials=N_TRIALS, show_progress_bar=False, callbacks=[print_progress])
            
            print(f"Best MAE: {study.best_value:.4f}")
            
            # Train final model
            best_model = get_model_with_params(model_name, study.best_params)
            best_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = best_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"Test - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            # Save model
            joblib.dump({
                'model': best_model,
                'model_name': model_name,
                'best_params': study.best_params,
                'performance': {'mae': mae, 'rmse': rmse, 'r2': r2},
                'y_pred': y_pred
            }, f'{output_dir}/models/{model_name}_model.joblib')
            
            results.append({
                'Model': model_name,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'Best_Params': str(study.best_params)
            })
            
            del study
            gc.collect()
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Save results
    results_df = pd.DataFrame(results).sort_values('MAE')
    results_df.to_csv(f'{output_dir}/individual_models_results.csv', index=False)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(results_df.to_string(index=False))

def optimize_random_forest(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': RANDOM_STATE, 'n_jobs': -1
    }
    model = RandomForestRegressor(**params)
    return -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1).mean()

def optimize_extra_trees(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': RANDOM_STATE, 'n_jobs': -1
    }
    model = ExtraTreesRegressor(**params)
    return -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1).mean()

def optimize_xgboost(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'random_state': RANDOM_STATE, 'n_jobs': -1, 'tree_method': 'hist'
    }
    model = xgb.XGBRegressor(**params)
    return -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1).mean()

def optimize_lightgbm(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1
    }
    model = lgb.LGBMRegressor(**params)
    return -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1).mean()

def optimize_catboost(trial, X, y):
    params = {
        'iterations': trial.suggest_int('iterations', 50, 500),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_state': RANDOM_STATE, 'verbose': 0, 'thread_count': -1
    }
    model = CatBoostRegressor(**params)
    return -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1).mean()

def optimize_svr(trial, X, y):
    params = {
        'C': trial.suggest_float('C', 0.1, 100, log=True),
        'epsilon': trial.suggest_float('epsilon', 0.001, 1, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
    }
    if params['kernel'] == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
    model = SVR(**params)
    return -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1).mean()

def optimize_knn(trial, X, y):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'p': trial.suggest_int('p', 1, 3),
        'leaf_size': trial.suggest_int('leaf_size', 10, 50),
        'n_jobs': -1
    }
    model = KNeighborsRegressor(**params)
    return -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1).mean()

def optimize_mlp(trial, X, y):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_layers = tuple([trial.suggest_int(f'n_units_l{i}', 50, 300) for i in range(n_layers)])
    params = {
        'hidden_layer_sizes': hidden_layers,
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.01, log=True),
        'max_iter': 1000, 'early_stopping': True, 'validation_fraction': 0.1,
        'n_iter_no_change': 50, 'random_state': RANDOM_STATE
    }
    model = MLPRegressor(**params)
    return -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1).mean()

def optimize_ridge(trial, X, y):
    params = {
        'alpha': trial.suggest_float('alpha', 0.01, 100, log=True),
        'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr']),
        'random_state': RANDOM_STATE
    }
    model = Ridge(**params)
    return -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1).mean()

def optimize_kernel_ridge(trial, X, y):
    params = {
        'alpha': trial.suggest_float('alpha', 0.01, 10, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'polynomial']),
        'gamma': trial.suggest_float('gamma', 0.001, 1, log=True)
    }
    if params['kernel'] == 'polynomial':
        params['degree'] = trial.suggest_int('degree', 2, 5)
    model = KernelRidge(**params)
    return -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1).mean()

def get_model_with_params(model_name, params):
    if model_name == 'MLP':
        mlp_params = params.copy()
        n_layers = mlp_params.pop('n_layers')
        hidden_layers = tuple([mlp_params.pop(f'n_units_l{i}') for i in range(n_layers)])
        mlp_params['hidden_layer_sizes'] = hidden_layers
        return MLPRegressor(**mlp_params, max_iter=1000, early_stopping=True, 
                           validation_fraction=0.1, n_iter_no_change=50, random_state=RANDOM_STATE)
    
    models = {
        'Random_Forest': lambda: RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1),
        'Extra_Trees': lambda: ExtraTreesRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost': lambda: xgb.XGBRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1, tree_method='hist'),
        'LightGBM': lambda: lgb.LGBMRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        'CatBoost': lambda: CatBoostRegressor(**params, random_state=RANDOM_STATE, verbose=0, thread_count=-1),
        'SVR': lambda: SVR(**params),
        'KNN': lambda: KNeighborsRegressor(**params, n_jobs=-1),
        'Ridge': lambda: Ridge(**params, random_state=RANDOM_STATE),
        'Kernel_Ridge': lambda: KernelRidge(**params)
    }
    return models[model_name]()

def train_ensemble_models(INPUT_CSV, INPUT_DIR, BANDGAP_MIN, BANDGAP_MAX, NUMBER_OF_MODELS_LIST, RANDOM_STATE=42):
    print("="*80)
    print("ENSEMBLE MODEL TRAINING")
    print("="*80)
    
    # ===== FIX: Load split configuration and recreate the same split =====
    split_data = joblib.load(f'{INPUT_DIR}/models/scaler_and_split.joblib')
    scaler = split_data['scaler']
    TEST_SIZE = split_data['test_size']
    RANDOM_STATE = split_data['random_state']
    
    # Load original data and prepare
    df = pd.read_csv(INPUT_CSV)
    bandgap_col = 'bandgap' if 'bandgap' in df.columns else 'BandGap'
    df_filtered = df[(df[bandgap_col] >= BANDGAP_MIN) & (df[bandgap_col] <= BANDGAP_MAX)].dropna()
    
    X = df_filtered.iloc[:, 2:].values.astype(np.float32)
    y = df_filtered[bandgap_col].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    
    # CRITICAL FIX: Recreate the SAME split as individual models
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    # ===== END FIX =====
    
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Verify we have the same split
    assert np.allclose(X_test, split_data['X_test']), "Test set mismatch!"
    assert np.allclose(y_test, split_data['y_test']), "Test labels mismatch!"
    print("✓ Split verification passed - using same data as individual models")
    
    del df, df_filtered, X, X_scaled
    gc.collect()
    
    # Load individual model results
    results_df = pd.read_csv(f'{INPUT_DIR}/individual_models_results.csv')
    results_df = results_df.sort_values('MAE')
    
    all_ensemble_results = []
    
    for n_models in NUMBER_OF_MODELS_LIST:
        print(f"\n{'='*60}\nBuilding ensembles with top {n_models} models\n{'='*60}")
        
        top_models = results_df.head(n_models)['Model'].tolist()
        print(f"Selected models: {top_models}")
        
        # Load individual models
        estimators = []
        for model_name in top_models:
            model_data = joblib.load(f'{INPUT_DIR}/models/{model_name}_model.joblib')
            estimators.append((model_name, model_data['model']))
        
        # 1. Weighted Voting Ensemble
        print(f"\nTraining Weighted_Voting_n{n_models}...")
        
        val_predictions = np.array([model.predict(X_test) for _, model in estimators])
        
        def weighted_mae(weights):
            weights = weights / weights.sum()
            weighted_pred = np.sum(val_predictions * weights[:, None], axis=0)
            return mean_absolute_error(y_test, weighted_pred)
        
        result = minimize(weighted_mae, np.ones(n_models)/n_models, method='SLSQP',
                         bounds=[(0, 1)]*n_models, constraints={'type': 'eq', 'fun': lambda w: w.sum()-1})
        
        optimal_weights = result.x
        print(f"Optimal weights: {dict(zip(top_models, optimal_weights))}")
        
        weighted_voting = VotingRegressor(estimators=estimators, weights=optimal_weights, n_jobs=-1)
        weighted_voting.fit(X_train, y_train)
        y_pred_weighted = weighted_voting.predict(X_test)
        
        mae_w = mean_absolute_error(y_test, y_pred_weighted)
        rmse_w = np.sqrt(mean_squared_error(y_test, y_pred_weighted))
        r2_w = r2_score(y_test, y_pred_weighted)
        
        print(f"MAE: {mae_w:.4f}, RMSE: {rmse_w:.4f}, R²: {r2_w:.4f}")
        
        weighted_name = f'Weighted_Voting_n{n_models}'
        joblib.dump({
            'model': weighted_voting,
            'model_name': weighted_name,
            'estimators': top_models,
            'weights': optimal_weights.tolist(),
            'performance': {'mae': mae_w, 'rmse': rmse_w, 'r2': r2_w},
            'y_pred': y_pred_weighted
        }, f'{INPUT_DIR}/models/{weighted_name}_model.joblib')
        
        all_ensemble_results.append({
            'Model': weighted_name,
            'MAE': mae_w,
            'RMSE': rmse_w,
            'R²': r2_w
        })
        
        # 2. Stacking Ensemble
        print(f"\nTraining Stacking_n{n_models}...")
        
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                                             random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
            n_jobs=-1
        )
        stacking.fit(X_train, y_train)
        y_pred_stacking = stacking.predict(X_test)
        
        mae_s = mean_absolute_error(y_test, y_pred_stacking)
        rmse_s = np.sqrt(mean_squared_error(y_test, y_pred_stacking))
        r2_s = r2_score(y_test, y_pred_stacking)
        
        print(f"MAE: {mae_s:.4f}, RMSE: {rmse_s:.4f}, R²: {r2_s:.4f}")
        
        stacking_name = f'Stacking_n{n_models}'
        joblib.dump({
            'model': stacking,
            'model_name': stacking_name,
            'estimators': top_models,
            'performance': {'mae': mae_s, 'rmse': rmse_s, 'r2': r2_s},
            'y_pred': y_pred_stacking
        }, f'{INPUT_DIR}/models/{stacking_name}_model.joblib')
        
        all_ensemble_results.append({
            'Model': stacking_name,
            'MAE': mae_s,
            'RMSE': rmse_s,
            'R²': r2_s
        })
        
        gc.collect()
    
    # Save ensemble results
    ensemble_df = pd.DataFrame(all_ensemble_results).sort_values('MAE')
    ensemble_df.to_csv(f'{INPUT_DIR}/ensemble_models_results.csv', index=False)
    
    print("\n" + "="*80)
    print("ENSEMBLE TRAINING COMPLETED")
    print("="*80)
    print(ensemble_df.to_string(index=False))


#######################################################################################################################
#                                        Regression model inference
#######################################################################################################################

def predict_Bandgap(INPUT_DIR, INPUT_CSV, MODEL_FILE, OUTPUT_CSV):
    print("="*80)
    print("BAND GAP PREDICTION INFERENCE")
    print("="*80)
    
    # Load model
    model_path = f'{INPUT_DIR}/models/{MODEL_FILE}'
    print(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    model_name = model_data['model_name']
    performance = model_data['performance']
    
    print(f"✓ Model loaded: {model_name}")
    print(f"  Test MAE: {performance['mae']:.4f}")
    print(f"  Test RMSE: {performance['rmse']:.4f}")
    print(f"  Test R²: {performance['r2']:.4f}")
    print("-"*80)
    
    # Load scaler
    split_data = joblib.load(f'{INPUT_DIR}/models/scaler_and_split.joblib')
    scaler = split_data['scaler']
    
    # Load input data
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")
    
    df = pd.read_csv(INPUT_CSV)
    print(f"\n✓ Input data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    formulas = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values
    
    print(f"  Sample formulas:")
    for i in range(min(3, len(formulas))):
        print(f"    {i+1}. {formulas[i]}")
    
    # Check data quality
    print("\n" + "-"*80)
    print("CHECKING DATA QUALITY")
    print("-"*80)
    
    valid_indices = []
    skipped_rows = []
    
    for i in range(len(features)):
        row = features[i]
        if np.any(np.isnan(row)) or np.any(np.isinf(row)):
            skipped_rows.append({'index': i, 'formula': formulas[i], 'reason': 'NaN or Inf'})
        else:
            valid_indices.append(i)
    
    print(f"✓ Valid rows: {len(valid_indices)}")
    
    if skipped_rows:
        print(f"⚠ Skipped rows: {len(skipped_rows)}")
        for skip in skipped_rows[:5]:
            print(f"  Row {skip['index']}: {skip['formula']} - {skip['reason']}")
        if len(skipped_rows) > 5:
            print(f"  ... and {len(skipped_rows) - 5} more")
    
    if len(valid_indices) == 0:
        raise ValueError("No valid rows found!")
    
    # Run predictions
    print("\n" + "-"*80)
    print("RUNNING PREDICTIONS")
    print("-"*80)
    
    valid_features = features[valid_indices]
    valid_formulas = formulas[valid_indices]
    
    X_scaled = scaler.transform(valid_features)
    predictions = model.predict(X_scaled)
    
    print(f"✓ Predictions completed for {len(predictions)} samples")
    
    # Statistics
    print(f"\nPREDICTION STATISTICS")
    print("-"*80)
    print(f"  Average predicted band gap: {np.mean(predictions):.4f} eV")
    print(f"  Min predicted band gap: {np.min(predictions):.4f} eV")
    print(f"  Max predicted band gap: {np.max(predictions):.4f} eV")
    print(f"  Std deviation: {np.std(predictions):.4f} eV")
    
    # Create output
    output_df = pd.DataFrame({
        'formula': valid_formulas,
        'predicted_bandgap': predictions
    })
    
    #output_df = output_df.sort_values('predicted_bandgap').reset_index(drop=True)
    
    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Predictions saved to: {OUTPUT_CSV}")
    
    if skipped_rows:
        skipped_csv = OUTPUT_CSV.replace('.csv', '_skipped_rows.csv')
        skipped_df = pd.DataFrame(skipped_rows)
        skipped_df.to_csv(skipped_csv, index=False)
        print(f"✓ Skipped rows saved to: {skipped_csv}")
    
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    print(output_df.head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETED")
    print("="*80)
    print(f"Total predictions: {len(output_df)}")
    print(f"Output file: {OUTPUT_CSV}")


