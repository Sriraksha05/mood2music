# app/streamlit_app.py
import sys, os
# add project root to path so `src` can be imported when Streamlit runs from app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.emotion_model import load_emotion_model, predict_emotion_from_image
from src.text_model import load_text_model, predict_text_mood
from src.recommender import recommend_by_mood

st.set_page_config(page_title="Mood → Music", layout='centered')

@st.cache_resource
def load_models():
    em = load_emotion_model('models/emotion_model.h5')
    tm = load_text_model('models/text_model')  # if not present, HF model will be downloaded (first run)
    return em, tm

em_model, txt_model = load_models()

st.title("Mood → Music Generator")

mode = st.radio("Choose input type:", ["Image", "Text"])

languages = st.multiselect("Select preferred languages:", ["Kannada", "Hindi", "English"], default=["English"])

if mode == "Image":
    uploaded = st.file_uploader("Upload a face photo", type=['jpg','png'])
    if uploaded is not None:
        st.image(uploaded, width=240)
        with st.spinner("Detecting mood..."):
            mood = predict_emotion_from_image(em_model, uploaded)
        st.write(f"**Detected mood:** {mood}")
        recs = recommend_by_mood(mood, languages, n_per_lang=2)
        if not recs:
            st.info("No recommendations found for selected languages/mood.")
        for r in recs:
            if r.get('album_image_url'):
                st.image(r.get('album_image_url'), width=100)
            track = r.get('track') or 'Unknown'
            artist = r.get('artist') or 'Unknown'
            url = r.get('preview_url') or r.get('url') or ''
            if url:
                st.markdown(f"**{track}** — {artist}  \n[{url}]({url})")
            else:
                st.write(f"**{track}** — {artist}")
else:
    txt = st.text_area("Describe how you feel (one sentence)")
    if st.button("Recommend"):
        with st.spinner("Detecting mood from text..."):
            mood = predict_text_mood(txt_model, txt)
        st.write(f"**Detected mood:** {mood}")
        recs = recommend_by_mood(mood, languages, n_per_lang=2)
        if not recs:
            st.info("No recommendations found for selected languages/mood.")
        for r in recs:
            if r.get('album_image_url'):
                st.image(r.get('album_image_url'), width=100)
            track = r.get('track') or 'Unknown'
            artist = r.get('artist') or 'Unknown'
            url = r.get('preview_url') or r.get('url') or ''
            if url:
                st.markdown(f"**{track}** — {artist}  \n[{url}]({url})")
            else:
                st.write(f"**{track}** — {artist}")
