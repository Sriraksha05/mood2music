# src/recommender.py
from typing import List, Dict, Any
from .spotify_client import get_spotify_client
import pandas as pd
import os
import random

CSV_PATH_DEFAULT = 'data/spotify_features.csv'

# reuse earlier functions (mood canonicalization & CSV/spotify recommenders)
MOOD_CANON = {
    'happy':'happy','joy':'happy','happiness':'happy',
    'sad':'sad','sadness':'sad',
    'angry':'angry','anger':'angry',
    'calm':'calm','neutral':'neutral',
    'energetic':'energetic','excited':'energetic',
    'surprise':'surprise','fear':'fear','disgust':'disgust'
}

# Maps synonyms and variations to canonical mood names
MOOD_CAN = {
    "happy": "joy",
    "joyful": "joy",
    "joy": "joy",
    "sad": "sadness",
    "depressed": "sadness",
    "angry": "anger",
    "mad": "anger",
    "fear": "fear",
    "scared": "fear",
    "surprised": "surprise",
    "shocked": "surprise",
    "neutral": "neutral"
}


def canonicalize_mood(m):
    if not m: return 'neutral'
    return MOOD_CAN.get(m.strip().lower(), m.strip().lower())

def _load_csv(path=CSV_PATH_DEFAULT):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if 'language' in df.columns:
        df['language'] = df['language'].astype(str).str.strip().str.lower()
    return df

def recommend_by_mood_spotify(mood: str, languages: List[str], n_per_lang: int = 2, sp=None):
    mood_c = canonicalize_mood(mood)
    langs = [l.strip() for l in languages if l and l.strip()]
    if not langs:
        langs = ['english']
    if sp is None:
        sp = get_spotify_client()
    recs = {}
    for lang in langs:
        q = f"{mood_c} {lang}"
        try:
            results = sp.search(q=q, type='track', limit=max(10, n_per_lang*3))
            items = results.get('tracks', {}).get('items', [])
        except Exception:
            items = []
        out = []
        seen = set()
        for it in items:
            tid = it.get('id')
            if not tid or tid in seen: continue
            seen.add(tid)
            out.append({
                'track': it.get('name'),
                'artist': ', '.join([a.get('name') for a in it.get('artists', [])]),
                'id': tid,
                'preview_url': it.get('preview_url'),
                'album_image_url': (it.get('album',{}).get('images') or [None])[0] and it.get('album',{}).get('images')[0].get('url'),
                'language': lang
            })
            if len(out) >= n_per_lang:
                break
        recs[lang] = out
    return recs

# CSV fallback similar to earlier; returns mapping lang->list
def recommend_by_mood_csv(mood: str, languages: List[str], n_per_lang: int = 2, csv_path=CSV_PATH_DEFAULT):
    mood_c = canonicalize_mood(mood)
    df = _load_csv(csv_path)
    if df.empty:
        return {lang: [] for lang in languages}
    results = {}
    for lang in languages:
        lang_l = lang.strip().lower()
        subset = df.copy()
        if 'language' in subset.columns:
            subset = subset[subset['language'].astype(str).str.lower() == lang_l]
        # basic heuristics using valence/energy if present
        if 'valence' in subset.columns and 'energy' in subset.columns:
            if mood_c == 'happy':
                subset = subset[(subset['valence']>=0.6) & (subset['energy']>=0.5)]
            elif mood_c == 'sad':
                subset = subset[(subset['valence']<=0.35) & (subset['energy']<=0.45)]
            elif mood_c == 'angry':
                subset = subset[(subset['energy']>=0.7)]
        # pick random n_per_lang
        if len(subset)==0:
            results[lang] = []
            continue
        chosen = subset.sample(n=min(n_per_lang, len(subset)), random_state=random.randint(0,9999))
        results[lang] = [{
            'track': r.get('track'),
            'artist': r.get('artist'),
            'id': r.get('id'),
            'preview_url': r.get('preview_url') if 'preview_url' in r else None,
            'album_image_url': r.get('album_image_url') if 'album_image_url' in r else None,
            'language': lang
        } for _, r in chosen.iterrows()]
    return results

# --- wrapper used by Streamlit app ---
def recommend_by_mood(mood: str, languages: List[str], n_per_lang: int = 2, prefer: str = 'spotify', csv_path: str = CSV_PATH_DEFAULT, client_id: str = None, client_secret: str = None):
    langs = [l for l in languages if l and l.strip()]
    if not langs:
        langs = ['english']
    if prefer == 'spotify':
        try:
            sp = get_spotify_client(client_id, client_secret)
            sp_recs = recommend_by_mood_spotify(mood, langs, n_per_lang, sp=sp)
            # fallback for languages that had no results
            missing = [lang for lang,lst in sp_recs.items() if not lst]
            if missing:
                csv_recs = recommend_by_mood_csv(mood, missing, n_per_lang, csv_path)
                for lang in missing:
                    sp_recs[lang] = csv_recs.get(lang, [])
            # flatten to list for streamlit iteration
            flat = []
            for lang in langs:
                for item in sp_recs.get(lang, []):
                    flat.append(item)
            return flat
        except Exception:
            # fallback to csv entirely
            csv_recs_map = recommend_by_mood_csv(mood, langs, n_per_lang, csv_path)
            flat = []
            for lang in langs:
                for item in csv_recs_map.get(lang, []):
                    flat.append(item)
            return flat
    else:
        csv_recs_map = recommend_by_mood_csv(mood, langs, n_per_lang, csv_path)
        flat = []
        for lang in langs:
            for item in csv_recs_map.get(lang, []):
                flat.append(item)
        return flat
