# Wine Quality Prediction (Assignment 2 - Machine Learning Application)

Predicts the quality score of red wine from 11 chemical properties (regression).
Models: Linear Regression vs Random Forest. Optional Streamlit interface included.

## Dataset
UCI Wine Quality (red): https://archive.ics.uci.edu/dataset/186/wine+quality
Direct CSV: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
A copy is already in `data/winequality-red.csv`. If it is missing, `train_model.py` downloads it.

## Steps
1. Install Python 3.9+ and open a terminal in this folder.
2. (Optional) create a virtual environment:
   `python -m venv venv`, then `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. Install libraries: `pip install -r requirements.txt`
4. Open `train_model.py` and edit STUDENT_NAME, ROLL_NUMBER, COLLEGE at the top.
5. Train and evaluate: `python train_model.py`
   - Plots, trained model and metrics go to `outputs/`
   - The Word report is created at `outputs/Wine_Quality_Report.docx`
6. Run the web app: `streamlit run app.py` (opens in your browser)

## Files
- `train_model.py` - full pipeline + auto-generated report
- `app.py` - Streamlit interface
- `requirements.txt` - libraries
- `data/` - dataset
- `outputs/` - plots, model, metrics, report
