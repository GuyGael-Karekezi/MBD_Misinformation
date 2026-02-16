# MBD Multimodal Misinformation

## Run locally

```bash
pip install -r requirements.txt
streamlit run demo/app.py
```

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repo.
2. Open https://share.streamlit.io and create a new app.
3. Select:
   - Repository: your repo
   - Branch: your deployment branch
   - Main file path: `demo/app.py`
4. Deploy.

The root `requirements.txt` points to `demo/requirements.txt`, so dependencies are installed automatically.
