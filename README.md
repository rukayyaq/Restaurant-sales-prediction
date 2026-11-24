# Restaurant Sales Prediction

Flask-based machine learning project for predicting restaurant demand (`quantity_sold`).

## Project structure

- `app.py` - Flask backend
- `train_model.py` - model training
- `model.pkl` - generated trained model
- `templates/` - HTML pages
- `static/` - CSS and JavaScript
- `requirements.txt` - dependencies

## Setup

1. Create/activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Put `restaurant_sales_data.csv` in the project root.
4. Run:

```bash
python train_model.py
```

5. Start Flask:

```bash
python app.py
```

6. Open the local Flask address shown in the terminal.

## Important

The current model script is a starter pipeline. Before final submission, the categorical/date preprocessing, train/test strategy, model comparison, hyperparameter tuning and analytics should be finalized using the actual dataset.
