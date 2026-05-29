# Northern Colorado Water Right Outlook — Live Prototype

This Streamlit app is a live-data prototype for a public-facing water-right outlook.

## What it does

- Pulls live active administrative calls from Colorado DWR/CDSS REST services.
- Filters to WD1, WD3, and WD69.
- Excludes WD2, WD4, WD5, WD6, WD7, WD8, and WD9.
- Classifies current basin regime:
  - Free River
  - Mild Administration
  - Normal Administration
  - Senior Administration
  - Exceptional Administration
- Trains a prototype 30-day Random Forest regime classifier from the included historical snapshot.
- Displays a public-facing dashboard.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Optional CDSS API key

The app will try to run anonymously. For more reliable service, set a CDSS API key:

```bash
export CDSS_API_KEY="your-key"
streamlit run app.py
```

For Streamlit Cloud, add this in Secrets:

```toml
CDSS_API_KEY = "your-key"
```

## Important limitation

This is not an official forecast, legal opinion, or substitute for CDSS/DWR records.
