push:
	git push github && git push gitlab

train-start-untuned:
	poetry run python src/training_pipeline/training.py --scenario start --models base lasso lightgbm xgboost

train-start-tuned:
	poetry run python src/training_pipeline/training.py --scenario start --models lasso lightgbm xgboost \
	--tune_hyperparameters --hyperparameter_trials 15

train-end-untuned:
	poetry run python src/training_pipeline/training.py --scenario end --models base lasso lightgbm xgboost

train-end-tuned:
	poetry run python src/training_pipeline/training.py --scenario end --models xgboost lasso lightgbm \
 	--tune_hyperparameters --hyperparameter_trials 15

train-all: train-start-untuned train-end-untuned train-start-tuned train-end-tuned

run-frontend:
	poetry run streamlit run src/inference_pipeline/frontend.py