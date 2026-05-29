# Northern Colorado Water Right Outlook — Public Beta with CDSS API

This version uses:
- CDSS live active administrative calls when enabled
- Historical daily regime data as the model backbone
- Expert water-right rules
- Public-facing Poudre River landing page

## Optional API key

CDSS supports anonymous use but applies call/row limits. For reliability, set a Streamlit secret:

```toml
CDSS_API_KEY = "your-key-here"
```

The app sends the key using the `ApiKey` request header.

## Deploy

Upload all files to GitHub and deploy on Streamlit Cloud with `app.py`.
