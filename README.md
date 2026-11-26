# AppPulse — Early-Warning System & Dynamic Competitor Intelligence

Search any app on Google Play, pick 2–4 competitors, scrape live reviews, benchmark
Logistic Regression vs Random Forest (`random_state=42`), auto-select the model with the highest
**Recall**, compare competitors on a dashboard, triage new reviews live, and generate an executive report with **Gemini Flash**.


Copied from: Editing apppulse-competitor-intelligence/README.md at main · doniaimad22-hue/apppulse-competitor-intelligence - <https://github.com/doniaimad22-hue/apppulse-competitor-intelligence/edit/main/README.md>

Demo set: Uber Eats vs DoorDash, Grubhub, Deliveroo.
Live app: https://apppulse-doniaa.streamlit.app/
## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, `README.md` to GitHub.
2. Create the app on share.streamlit.io.
3. In **Settings → Secrets** add: `GEMINI_API_KEY = "your-key"`.

Never commit API keys.

## Pipeline
Search → Scrape → Clean + feature engineering (length, words, `!`, likes, failure keywords, caps ratio, hour)
→ target `critical = 1` if rating ≤ 2 → benchmark → dashboard → live triage → AI report + TXT/CSV export.
