train-all-tuned:
	poetry run python training.py --scenario start --model lasso --tune_hyperparameters --hyperparameter_trials 15
	poetry run python training.py --scenario end --model lasso --tune_hyperparameters --hyperparameter_trials 15
	poetry run python training.py --scenario start --model lightgbm --tune_hyperparameters --hyperparameter_trials 15
	poetry run python training.py --scenario end --model lightgbm --tune_hyperparameters --hyperparameter_trials 15
	poetry run python training.py --scenario start --model xgboost --tune_hyperparameters --hyperparameter_trials 15
	poetry run python training.py --scenario end --model xgboost --tune_hyperparameters --hyperparameter_trials 15


train-all-not-tuned:
	poetry run python training.py --scenario start --model base
	poetry run python training.py --scenario end --model base
	poetry run python training.py --scenario start --model lasso
	poetry run python training.py --scenario end --model lasso
	poetry run python training.py --scenario start --model lightgbm
	poetry run python training.py --scenario end --model lightgbm
	poetry run python training.py --scenario start --model xgboost
	poetry run python training.py --scenario end --model xgboost
