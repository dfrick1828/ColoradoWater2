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


## Current call selection

The live current-call display now uses the most recently set active call from the included districts, rather than the most severe active call.


## Same-day call-depth percentile

The landing-page "Today's condition" metric now compares the current call-depth/severity score against historical observations for the same day of year, rather than against all days in the record. If exact day-of-year history is sparse, the app falls back to a +/- 7 day seasonal window.


## Latest patch

- The default scenario/historical date now opens on the current calendar day.
- If the historical snapshot does not contain today's exact date, the app uses the most recent historical row with the same day-of-year as a model template.
- Removed the "Area covered by the Outlook" section from the landing page.


## Current call selection rule

The reported/current call is selected as the most senior active call in WD1 and WD3.
Selection uses lowest CDSS Priority Admin No, with Priority Date as a fallback.


## Current Poudre call selection rule

The public-facing current call is selected as the active Cache la Poudre call:
- WD3 records;
- prefer Water Source / Location Structure / Priority Structure fields containing "POUDRE";
- select the most recently set active Poudre call, using Priority Admin No as a tie-breaker.

This is intended to show the current Poudre call, such as the 11/20/1874 call, rather than the most senior call elsewhere in WD1/WD3.
