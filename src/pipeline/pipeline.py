"""
End-to-end ML pipeline.
Ingestion → Cleaning → Problem Detection → Split/Scale → Model Loop (GridSearch)
→ Evaluate → Plots → Champion Selection → Save.
"""
import importlib
from pathlib import Path

import pandas as pd

from src.preprocessing.preprocessing import load_data, clean_data, detect_problem_type, save_processed
from src.training.training import split_and_scale, train_with_gridsearch, predict_and_evaluate
from src.plots.plotting import (
    plot_correlation,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_decision_tree,
    plot_actual_vs_predicted,
    plot_residuals,
)
from src.utils.io_utils import generate_timestamp, save_model_artifact
from src.tracking.mlflow_tracker import MLflowTracker


class MLPipeline:
    def __init__(self, raw_data_path, target_column, output_dir='outputs', test_size=0.2, random_state=42, experiment_name='ml_pipeline'):
        self.raw_data_path = raw_data_path
        self.target_column = target_column
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.random_state = random_state
        self.timestamp = generate_timestamp()
        self.results = []
        self.champion = None
        self.tracker = MLflowTracker(experiment_name)

    def run(self):
        # 1. Ingestion
        print("Step 1: Loading data...")
        df = load_data(self.raw_data_path)

        # 2. Preprocessing
        print("Step 2: Cleaning data...")
        cleaned_df, report = clean_data(df)
        processed_path = self.output_dir / 'processed' / f'data_{self.timestamp}.csv'
        save_processed(cleaned_df, processed_path)

        # 3. Detect problem type
        print("Step 3: Detecting problem type...")
        y = cleaned_df[self.target_column]
        problem_type = detect_problem_type(y)
        print(f"  -> {problem_type}")

        # 4. Prepare features
        feature_cols = [c for c in cleaned_df.columns if c != self.target_column]
        X = cleaned_df[feature_cols]

        # 5. Split and scale
        print("Step 4: Splitting and scaling...")
        X_train, X_test, y_train, y_test, scaler = split_and_scale(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=(problem_type == "classification"),
        )

        # 6. Correlation plot
        print("Step 5: Generating correlation plot...")
        corr_path = self.output_dir / 'plots' / f'correlation_{self.timestamp}.png'
        plot_correlation(cleaned_df, feature_cols, self.target_column, corr_path)

        # 7. Model loop
        print("Step 6: Training models...")
        models_config = self._get_models_config(problem_type)

        for model_cfg in models_config:
            run_name = f"{model_cfg['name']}_{self.timestamp}"
            print(f"\n  Training {model_cfg['name']}...")

            with self.tracker.start_run(run_name=run_name):
                # Log pipeline-level params
                self.tracker.log_params({
                    'model_name': model_cfg['name'],
                    'problem_type': problem_type,
                    'test_size': self.test_size,
                    'random_state': self.random_state,
                    'n_features': len(feature_cols),
                    'n_samples': len(X),
                })

                model_class = self._import_model(model_cfg['module'], model_cfg['class'])

                # GridSearchCV if param_grid exists, else simple fit
                if model_cfg['param_grid']:
                    best_model, best_params = train_with_gridsearch(
                        model_class(), model_cfg['param_grid'], X_train, y_train, cv=5
                    )
                else:
                    best_model = model_class()
                    best_model.fit(X_train, y_train)
                    best_params = {}

                # Log best params
                self.tracker.log_params(best_params)

                # Evaluate
                y_pred, metrics = predict_and_evaluate(best_model, X_test, y_test, problem_type)

                # Log metrics
                self.tracker.log_metrics(metrics)

                # Plots
                plot_paths = self._generate_plots(
                    best_model, model_cfg['name'], X, y_test, y_pred, problem_type, feature_cols
                )

                # Log plot artifacts
                for plot_path in plot_paths.values():
                    self.tracker.log_artifact(plot_path)

                # Log model
                self.tracker.log_model(best_model, "model")

                # Store result
                result = {
                    'model_name': model_cfg['name'],
                    'problem_type': problem_type,
                    'best_params': best_params,
                    'metrics': metrics,
                    'plots': plot_paths,
                    'model': best_model,
                }
                self.results.append(result)

                # Print summary (skip long classification report)
                summary = {k: round(v, 4) if isinstance(v, (int, float)) else v
                          for k, v in metrics.items() if k != 'classification_report'}
                print(f"    Metrics: {summary}")

            self.tracker.end_run()

        # 8. Champion selection
        print("\nStep 7: Selecting champion...")
        self._select_champion(problem_type)

        if self.champion:
            champion_path = self.output_dir / 'models' / f"champion_{self.champion['model_name']}_{self.timestamp}.pkl"
            save_model_artifact(self.champion['model'], champion_path)
            print(f"  Champion: {self.champion['model_name']}")
            print(f"  Primary metric: {self.champion['metrics'].get('f1') or self.champion['metrics'].get('r2')}")
            print(f"  Saved to: {champion_path}")

        return self.results, self.champion

    def _get_models_config(self, problem_type):
        if problem_type == 'classification':
            return [
                {
                    'name': 'KNN',
                    'module': 'sklearn.neighbors',
                    'class': 'KNeighborsClassifier',
                    'param_grid': {
                        'n_neighbors': [1, 3, 5, 7, 9],
                        'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'manhattan']
                    }
                },
                {
                    'name': 'DecisionTree',
                    'module': 'sklearn.tree',
                    'class': 'DecisionTreeClassifier',
                    'param_grid': {
                        'max_depth': [1, 3, 5, 7, 10],
                        'criterion': ['gini', 'entropy']
                    }
                },
                {
                    'name': 'RandomForest',
                    'module': 'sklearn.ensemble',
                    'class': 'RandomForestClassifier',
                    'param_grid': {
                        'n_estimators': [50, 100],
                        'max_depth': [5, 10],
                        'min_samples_split': [2, 5]
                    }
                },
                {
                    'name': 'GradientBoosting',
                    'module': 'sklearn.ensemble',
                    'class': 'GradientBoostingClassifier',
                    'param_grid': {
                        'n_estimators': [50, 100],
                        'max_depth': [3, 5],
                        'learning_rate': [0.05, 0.1]
                    }
                },
                {
                    'name': 'ExtraTrees',
                    'module': 'sklearn.ensemble',
                    'class': 'ExtraTreesClassifier',
                    'param_grid': {
                        'n_estimators': [50, 100],
                        'max_depth': [5, 10],
                        'min_samples_split': [2, 5]
                    }
                },
                {
                    'name': 'LogisticRegression',
                    'module': 'sklearn.linear_model',
                    'class': 'LogisticRegression',
                    'param_grid': {
                        'C': [0.1, 1.0, 10.0],
                        'solver': ['lbfgs']
                    }
                }
            ]
        else:
            return [
                {
                    'name': 'LinearRegression',
                    'module': 'sklearn.linear_model',
                    'class': 'LinearRegression',
                    'param_grid': {}
                },
                {
                    'name': 'Ridge',
                    'module': 'sklearn.linear_model',
                    'class': 'Ridge',
                    'param_grid': {
                        'alpha': [0.1, 1.0, 10.0]
                    }
                },
                {
                    'name': 'Lasso',
                    'module': 'sklearn.linear_model',
                    'class': 'Lasso',
                    'param_grid': {
                        'alpha': [0.1, 1.0, 10.0]
                    }
                },
                {
                    'name': 'ElasticNet',
                    'module': 'sklearn.linear_model',
                    'class': 'ElasticNet',
                    'param_grid': {
                        'alpha': [0.1, 1.0],
                        'l1_ratio': [0.2, 0.5, 0.8]
                    }
                },
                {
                    'name': 'RandomForest',
                    'module': 'sklearn.ensemble',
                    'class': 'RandomForestRegressor',
                    'param_grid': {
                        'n_estimators': [50, 100],
                        'max_depth': [5, 10]
                    }
                },
                {
                    'name': 'GradientBoosting',
                    'module': 'sklearn.ensemble',
                    'class': 'GradientBoostingRegressor',
                    'param_grid': {
                        'n_estimators': [50, 100],
                        'max_depth': [3, 5],
                        'learning_rate': [0.05, 0.1]
                    }
                }
            ]

    def _import_model(self, module_name, class_name):
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def _generate_plots(self, model, model_name, X, y_test, y_pred, problem_type, feature_cols):
        plots = {}
        plot_dir = self.output_dir / 'plots'

        if problem_type == 'classification':
            cm_path = plot_dir / f'confusion_{model_name}_{self.timestamp}.png'
            plot_confusion_matrix(y_test, y_pred, cm_path)
            plots['confusion_matrix'] = cm_path

            if hasattr(model, 'feature_importances_'):
                fi_path = plot_dir / f'feature_importance_{model_name}_{self.timestamp}.png'
                plot_feature_importance(model, feature_cols, fi_path)
                plots['feature_importance'] = fi_path

            if model_name == 'DecisionTree' and hasattr(model, 'tree_'):
                dt_path = plot_dir / f'decision_tree_{model_name}_{self.timestamp}.png'
                plot_decision_tree(model, feature_cols, model.classes_, dt_path)
                plots['decision_tree'] = dt_path

        elif problem_type == 'regression':
            avp_path = plot_dir / f'actual_vs_pred_{model_name}_{self.timestamp}.png'
            plot_actual_vs_predicted(y_test, y_pred, avp_path)
            plots['actual_vs_predicted'] = avp_path

            residuals = y_test - y_pred
            resid_path = plot_dir / f'residuals_{model_name}_{self.timestamp}.png'
            plot_residuals(y_pred, residuals, resid_path)
            plots['residuals'] = resid_path

        return plots

    def _select_champion(self, problem_type):
        if not self.results:
            return

        primary = 'f1' if problem_type == 'classification' else 'r2'
        self.champion = max(self.results, key=lambda r: r['metrics'][primary])