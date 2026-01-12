# Credit-card-fraud-detection

Simple credit card fraud detection project using dataset from `dataset/`.

## Quick start

1. Create and activate a virtual environment

   python -m venv venv
   # Windows
   .\venv\Scripts\activate

2. Install requirements

   pip install -r requirement.txt

3. Train model and run app

   python train_model.py
   streamlit run app_streamlit.py

## Notes
- Large data and model artifacts are excluded from the repository via `.gitignore`. If you want them tracked, remove `dataset/` and `*.pkl` from `.gitignore`. 