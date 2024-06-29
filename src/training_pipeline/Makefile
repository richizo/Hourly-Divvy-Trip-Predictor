train-all-tuned:
	python training.py --scenario start --model lasso --tune_hyperparameters --hyperparameter_trials 15
	python training.py --scenario end --model lasso --tune_hyperparameters --hyperparameter_trials 15
	python training.py --scenario start --model lightgbm --tune_hyperparameters --hyperparameter_trials 15
	python training.py --scenario end --model lightgbm --tune_hyperparameters --hyperparameter_trials 15
	python training.py --scenario start --model xgboost --tune_hyperparameters --hyperparameter_trials 15
	python training.py --scenario end --model xgboost --tune_hyperparameters --hyperparameter_trials 15


train-all-not-tuned:
	python training.py --scenario start --model base
	python training.py --scenario end --model base
	python training.py --scenario start --model lasso
	python training.py --scenario end --model lasso
	python training.py --scenario start --model lightgbm
	python training.py --scenario end --model lightgbm
	python training.py --scenario start --model xgboost
	python training.py --scenario end --model xgboost
