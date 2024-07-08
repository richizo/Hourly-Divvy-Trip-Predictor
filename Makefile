push:
	git push github && git push gitlab

train-start-untuned:
	poetry run python src/training_pipeline/training.py --scenario start --models base lasso lightgbm

train-start-tuned:
	poetry run python src/training_pipeline/training.py --scenario start --models base lasso lightgbm xgboost -- \
		   tune_hyperparameters --hyperparameter_trials 15

train-end-untuned:
	poetry run python src/training_pipeline/training.py --scenario end --models base lasso lightgbm xgboost

train-end-tuned:
	poetry run python src/training_pipeline/training.py --scenario end --models base lasso lightgbm xgboost -- \
		   tune_hyperparameters --hyperparameter_trials 15