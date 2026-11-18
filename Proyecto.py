from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Recomendación de cultivos", layout="wide")

BASE = Path(__file__).resolve().parent
DATA_ORIGINAL = BASE / "merged_crop_dataset.csv"
DATA = BASE / "merged_crop_dataset_es.csv"
MODEL = BASE / "crop_model.joblib"
CLUSTERS = BASE / "crop_clusters.joblib"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
NAMES = {"N": "Nitrógeno", "P": "Fósforo", "K": "Potasio",
         "temperature": "Temperatura", "humidity": "Humedad",
         "ph": "pH del suelo", "rainfall": "Lluvia"}
UNITS = {"N": "mg/kg", "P": "mg/kg", "K": "mg/kg", "temperature": "°C",
         "humidity": "%", "ph": "", "rainfall": "mm"}

# Nombres normalizados de los cultivos del dataset en español.
TRADUCCION_CULTIVOS = {
    "rice": "Arroz",
    "maize": "Maíz",
    "chickpea": "Garbanzo",
    "kidneybeans": "Frijol riñón",
    "pigeonpeas": "Guandú",
    "mothbeans": "Frijol moth",
    "mungbean": "Frijol mungo",
    "blackgram": "Frijol negro",
    "lentil": "Lenteja",
    "pomegranate": "Granada",
    "banana": "Banano",
    "mango": "Mango",
    "grapes": "Uvas",
    "watermelon": "Sandía",
    "muskmelon": "Melón",
    "apple": "Manzana",
    "orange": "Naranja",
    "papaya": "Papaya",
    "coconut": "Coco",
    "cotton": "Algodón",
    "jute": "Yute",
    "coffee": "Café",
    "potato": "Papa",
    "onion": "Cebolla",
    "sorghum": "Sorgo",
    "sunflower": "Girasol",
    "rapeseed": "Colza",
    "ragi": "Mijo dedo",
    "horsegram": "Frijol caballo",
    "wheat": "Trigo",
    "turmeric": "Cúrcuma",
    "sweetpotato": "Camote",
    "soybean": "Soya",
    "barley": "Cebada",
    "coriander": "Culantro",
    "garlic": "Ajo",
    "blackpepper": "Pimienta negra",
    "tomato": "Tomate",
    "brinjal": "Berenjena",
    "cardamom": "Cardamomo",
    "okra": "Quimbombó",
    "pineapple": "Piña",
    "cabbage": "Repollo",
    "cauliflower": "Coliflor",
    "jackfruit": "Yaca",
    "drumstick": "Moringa",
    "cucumber": "Pepino",
    "radish": "Rábano",
    "bottlegourd": "Calabaza botella",
    "bittergourd": "Melón amargo",
    "pumpkin": "Calabaza",
}

def traducir_cultivo(nombre):
    clave = str(nombre).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    return TRADUCCION_CULTIVOS.get(clave, str(nombre).strip())

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
.stApp{background:#f2f8fa;color:#173746;font-family:'DM Sans',sans-serif}
.block-container{max-width:1400px;padding-top:1.5rem;padding-bottom:3rem}
h1,h2,h3{font-family:'Manrope',sans-serif!important;color:#173746!important}
[data-testid="stSidebar"]{background:#bfeaf5}
[data-testid="stSidebar"] *{color:#164b5b!important}
[data-testid="stSidebar"] button,[data-testid="stSidebarCollapsedControl"] button{background:#bfeaf5!important;color:#164b5b!important;border-color:#78cfe5!important}
[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child{border-color:#8b5cf6!important}
[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) > div:first-child{background-color:#8b5cf6!important;border-color:#8b5cf6!important}
[data-testid="stSidebar"] input[type="radio"]{accent-color:#8b5cf6!important}
.stButton>button{background:#269fbd;color:white;border:0;border-radius:12px;min-height:2.8rem;font-weight:700}
.stButton>button:hover{background:#187f9a;color:white;border:0}

div[data-testid="stMetric"]{background:white;border:1px solid #dce9ed;border-radius:16px;padding:1rem}
div[data-testid="stMetricLabel"]{color:#65776c!important}
div[data-testid="stMetricValue"]{color:#173746!important}
.hero{position:relative;isolation:isolate;overflow:hidden;background:linear-gradient(90deg,rgba(12,63,83,.88),rgba(22,141,163,.66)),url('https://bloglatam.jacto.com/wp-content/uploads/2023/04/jacto_jactomexico_image_646-scaled.jpeg') center/cover no-repeat;padding:2.2rem;border-radius:25px;color:white;margin-bottom:1.5rem;min-height:230px;display:flex;flex-direction:column;justify-content:center}
.hero *{color:white!important}
.kicker{font-size:.72rem;letter-spacing:2px;font-weight:800;opacity:.82}
.hero h1{font-size:clamp(2.1rem,4vw,3.4rem)!important;line-height:1.08;margin:.65rem 0!important;color:white!important}
.hero p{font-size:1rem;max-width:760px;color:white!important;opacity:.92}
.panel{background:white;border:1px solid #dce9ed;border-radius:18px;padding:1.2rem;height:100%}
.panel h3{font-size:1.05rem;margin-top:.4rem}
.panel p{color:#66776c;font-size:.9rem}
.kicker-green{color:#269fbd;font-size:.72rem;letter-spacing:1.5px;font-weight:800}
.note{background:#e8f4f7;border-left:4px solid #269fbd;padding:.9rem 1rem;border-radius:10px;color:#254c58}
.reco{background:white;border:1px solid #dce9ed;border-radius:18px;padding:1.2rem;min-height:190px}
.reco .rank{font-size:.7rem;font-weight:800;letter-spacing:1.2px;text-transform:uppercase;color:#66776c}
.reco .crop{font-family:'Manrope',sans-serif;font-size:1.5rem;font-weight:800;margin:.7rem 0 .2rem;color:#173746;overflow-wrap:anywhere}
.reco .score{font-size:1.8rem;font-weight:800}
.reco .track{height:7px;background:#edf1e9;border-radius:20px;overflow:hidden;margin-top:.6rem}
.reco .fill{height:100%;border-radius:20px}
.first{border-top:5px solid #56B4E9}.first .score{color:#1676A3}.first .fill{background:#56B4E9}
.second{border-top:5px solid #E69F00}.second .score{color:#9A6500}.second .fill{background:#E69F00}
.third{border-top:5px solid #CC79A7}.third .score{color:#994A78}.third .fill{background:#CC79A7}

[data-testid="stSelectbox"] label p{font-size:1.15rem!important;font-weight:700!important;color:#164b5b!important}
[data-testid="stSelectbox"] [data-baseweb="select"] > div{background:white!important;border:2px solid #269fbd!important;border-radius:12px!important;min-height:3.2rem;font-size:1.1rem;font-weight:600;box-shadow:0 3px 12px rgba(38,159,189,.18)}
[data-testid="stSelectbox"] [data-baseweb="select"] svg{fill:#269fbd!important;width:1.4rem;height:1.4rem}
button[data-baseweb="tab"]{height:auto;padding:.9rem 1.8rem}
button[data-baseweb="tab"] p{font-size:1.15rem!important;font-weight:700!important}
.profile{background:white;border:1px solid #dce9ed;border-radius:16px;padding:1rem;margin-bottom:.8rem}

.stat-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;margin:.5rem 0 1.3rem}
.stat-card{background:#fff;border:1px solid #dce9ed;border-radius:16px;padding:1.15rem 1.25rem;min-height:112px;box-shadow:0 3px 12px rgba(22,75,91,.05)}
.stat-label{color:#53685a!important;font-size:.86rem;font-weight:600;margin-bottom:.65rem}
.stat-value{color:#164b5b!important;font-family:'Manrope',sans-serif;font-size:2rem;line-height:1.1;font-weight:800}
@media(max-width:700px){.stat-grid{grid-template-columns:1fr}.stat-card{min-height:auto}}

.footer{color:#7a897f;font-size:.78rem;text-align:center;padding-top:2rem}
</style>
""", unsafe_allow_html=True)


def norm(value):
    return str(value).replace("\ufeff", "").replace("\xa0", " ").strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def read_data():
    if not DATA_ORIGINAL.exists():
        raise FileNotFoundError(
            "No se encontró merged_crop_dataset.csv. Guardá el archivo original "
            "en la misma carpeta que Proyecto.py."
        )
    try:
        df = pd.read_csv(DATA_ORIGINAL, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_ORIGINAL, sep=None, engine="python", encoding="latin-1")
    aliases = {
        "n":"N", "nitrogen":"N", "nitrogeno":"N", "p":"P", "phosphorus":"P",
        "fosforo":"P", "k":"K", "potassium":"K", "potasio":"K",
        "temperature":"temperature", "temperatura":"temperature",
        "humidity":"humidity", "humedad":"humidity", "ph":"ph",
        "rainfall":"rainfall", "lluvia":"rainfall", "label":"label",
        "crop":"label", "cultivo":"label"
    }
    df = df.rename(columns={c: aliases[norm(c)] for c in df.columns if norm(c) in aliases})
    required = FEATURES + ["label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Faltan columnas: " + ", ".join(missing) + ". Se necesitan N, P, K, temperature, humidity, ph, rainfall y label.")
    df = df[required].copy()
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["label"] = df["label"].astype(str).str.strip().map(traducir_cultivo)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    df = df[df["label"] != ""]
    if df.empty or df["label"].nunique() < 2:
        raise ValueError("El dataset debe contener datos válidos de al menos dos cultivos.")

    # Guarda una copia con los nombres de los cultivos traducidos, sin alterar el CSV original.
    df.to_csv(DATA, index=False, encoding="utf-8-sig")
    return df.reset_index(drop=True)


def train_model():
    df = read_data()
    X, y = df[FEATURES], df["label"]
    counts = y.value_counts()
    stratify = y if len(df) >= 10 and y.nunique() > 1 and counts.min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=stratify)
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
    model.fit(Xtr, ytr)
    accuracy = accuracy_score(yte, model.predict(Xte))
    model.fit(X, y)
    pack = {"modelo": model, "cultivos": sorted(y.unique().tolist()),
            "filas_dataset": len(df), "exactitud_prueba": float(accuracy)}
    joblib.dump(pack, MODEL)
    return pack


@st.cache_resource(show_spinner=False)
def get_model():
    expected_crops = set(read_data()["label"].unique())
    if MODEL.exists():
        saved = joblib.load(MODEL)
        saved_crops = set(str(x) for x in saved["modelo"].classes_)
        if saved_crops == expected_crops:
            return saved, False
    return train_model(), True


def train_clusters():
    df = read_data()
    by_crop = {}
    for crop, rows in df.groupby("label"):
        X = rows[FEATURES]
        if len(X) < 6:
            continue
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X)
        k = min(3, len(X) // 2)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        grouped = rows.copy()
        grouped["cluster"] = labels
        profiles = {}
        for cid in sorted(grouped["cluster"].unique()):
            g = grouped[grouped["cluster"] == cid]
            profiles[int(cid)] = {
                "cantidad": len(g),
                "promedios": g[FEATURES].mean().to_dict(),
                "minimos": g[FEATURES].min().to_dict(),
                "maximos": g[FEATURES].max().to_dict()
            }
        by_crop[crop] = {"perfiles": profiles, "total_muestras": len(rows)}
    pack = {"clusters": by_crop, "filas_dataset": len(df)}
    joblib.dump(pack, CLUSTERS)
    return pack


@st.cache_resource(show_spinner=False)
def get_clusters():
    expected = read_data()
    expected_crops = {
        str(crop) for crop, rows in expected.groupby("label") if len(rows) >= 6
    }
    if CLUSTERS.exists():
        saved = joblib.load(CLUSTERS)
        saved_crops = set(saved.get("clusters", {}).keys())
        if saved_crops == expected_crops:
            return saved, False
    return train_clusters(), True


with st.sidebar:
    st.markdown(
        """
        <div style="color:#164b5b;font-size:.78rem;font-weight:800;
                    letter-spacing:2px;margin:.5rem 0 1rem 0;">
            MENÚ PRINCIPAL
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "Seleccioná una sección",
        ["Inicio", "Recomendar un cultivo", "Perfiles de condiciones", "Sobre el modelo"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Explorá recomendaciones y condiciones de cultivo.")

st.markdown("""
<div class="hero">
 <div class="kicker">RECOMENDACIÓN DE CULTIVOS · DECISIONES CON DATOS</div>
 <h1>Encontrá el cultivo<br>que se ajusta a tus condiciones.</h1>
 <p>Una herramienta interactiva para explorar recomendaciones de cultivos y perfiles de suelo y clima a partir de datos.</p>
</div>
""", unsafe_allow_html=True)

try:
    with st.spinner("Preparando los modelos..."):
        pack, model_new = get_model()
        cluster_pack, clusters_new = get_clusters()
except Exception as e:
    st.error("No se pudieron preparar los modelos. Revisá el dataset y las columnas.")
    st.exception(e)
    st.stop()

model = pack["modelo"]
crop_profiles = cluster_pack["clusters"]

if model_new or clusters_new:
    st.success("Los modelos se prepararon y se guardaron. En próximas ejecuciones se reutilizarán.")

if page == "Inicio":
    st.markdown('<div class="kicker-green">PANEL GENERAL</div>', unsafe_allow_html=True)
    st.header("Tu espacio de exploración")
    st.markdown(
        f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Registros disponibles</div>
                <div class="stat-value">{pack['filas_dataset']:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Cultivos en el dataset</div>
                <div class="stat-value">{len(pack['cultivos'])}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Cultivos con perfiles</div>
                <div class="stat-value">{len(crop_profiles)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="panel"><div class="kicker-green">01 · CLASIFICACIÓN</div><h3>Recomendá un cultivo</h3><p>Ingresá siete variables del suelo y el clima para obtener un ranking de tres opciones.</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="kicker-green">02 · CLUSTERING</div><h3>Explorá condiciones</h3><p>Compará perfiles de muestras parecidas para un cultivo seleccionado.</p></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("### Cómo usar la herramienta")
    cols = st.columns(3)
    for col, title, desc in zip(cols,
        ["1. Ingresá los datos", "2. Revisá el ranking", "3. Compará perfiles"],
        ["Usá mediciones del suelo y el ambiente.", "Celeste, amarillo y rosado distinguen el orden de las opciones.", "Los grupos muestran patrones, no niveles de calidad."]):
        with col:
            st.markdown(f'<div class="panel"><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

    st.write("")
    publico, fuente = st.columns(2, gap="large")
    with publico:
        st.markdown(
            '<div class="panel"><div class="kicker-green">PÚBLICO META</div><h3>¿A quién está dirigida?</h3><p>Esta herramienta está dirigida principalmente a profesionales y técnicos del sector agrícola, como ingenieros agrónomos, interesados en identificar cultivos que se adapten a las condiciones de los terrenos. Su objetivo es servir como apoyo en la toma de decisiones relacionadas con la selección de cultivos.</p></div>',
            unsafe_allow_html=True,
        )
    with fuente:
        st.markdown(
            '<div class="panel"><div class="kicker-green">FUENTE DE DATOS</div><h3>Crop Recommendation</h3><p>El conjunto de datos utilizado para desarrollar el modelo está disponible en Kaggle.</p><a href="https://www.kaggle.com/datasets/sashankgarg23/crop-dataset" target="_blank" rel="noopener noreferrer">Consultar conjunto de datos en Kaggle</a></div>',
            unsafe_allow_html=True,
        )

elif page == "Recomendar un cultivo":
    st.markdown('<div class="kicker-green">01 · CLASIFICACIÓN</div>', unsafe_allow_html=True)
    st.header("¿Qué cultivo coincide con tu terreno?")
    st.write("Ingresá las condiciones conocidas y analizá las opciones que el modelo clasifica mejor.")
    with st.form("crop_form"):
        st.markdown("#### Nutrientes del suelo")
        c1, c2, c3 = st.columns(3)
        with c1:
            n = st.number_input("Nitrógeno (N) · mg/kg", min_value=0.0, max_value=2000.0, value=90.0, step=1.0)
        with c2:
            p = st.number_input("Fósforo (P) · mg/kg", min_value=0.0, max_value=2000.0, value=42.0, step=1.0)
        with c3:
            k = st.number_input("Potasio (K) · mg/kg", min_value=0.0, max_value=2000.0, value=43.0, step=1.0)
        st.markdown("#### Clima y suelo")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            temp = st.number_input("Temperatura (°C)", min_value=-10.0, max_value=70.0, value=25.0, step=.5)
        with c2:
            hum = st.number_input("Humedad (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        with c3:
            ph = st.number_input("pH del suelo", min_value=0.0, max_value=14.0, value=6.5, step=.1)
        with c4:
            rain = st.number_input("Lluvia (mm)", min_value=0.0, max_value=5000.0, value=100.0, step=1.0)
        submit = st.form_submit_button("Analizar mis condiciones", use_container_width=True, type="primary")
    if submit:
        sample = pd.DataFrame([[n, p, k, temp, hum, ph, rain]], columns=FEATURES)
        probs = model.predict_proba(sample)[0]
        classes = model.classes_
        top = np.argsort(probs)[::-1][:3]
        st.markdown("### Resultado del análisis")
        st.markdown(f'<div class="note"><strong>Primera opción del modelo:</strong> {str(classes[top[0]])}. El ranking refleja similitud con patrones aprendidos, no garantiza el éxito real de la cosecha.</div>', unsafe_allow_html=True)
        st.write("")
        cols = st.columns(3, gap="medium")
        styles = ["first", "second", "third"]
        names = ["1 · MAYOR PUNTUACIÓN", "2 · SEGUNDA OPCIÓN", "3 · TERCERA OPCIÓN"]
        semaforo = ["#2e9e4f", "#f2c200", "#d93025"]  # verde, amarillo, rojo
        for rank_position, (col, idx, style, label) in enumerate(zip(cols, top, styles, names)):
            score = float(probs[idx]) * 100
            width = min(100, max(0, score))
            crop = str(classes[idx])
            with col:
                st.markdown(
                    f'<div class="reco {style}" style="border-bottom:5px solid {semaforo[rank_position]}">'
                    f'<div class="rank">{label}</div>'
                    f'<div class="crop">{crop}</div>'
                    f'<div class="score">{score:.1f}%</div>'
                    f'<div class="track"><div class="fill" style="width:{width:.1f}%"></div></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        st.markdown("#### Resumen de datos ingresados")
        summary = pd.DataFrame({"Variable": [NAMES[x] for x in FEATURES],
                                "Valor": [n, p, k, temp, hum, ph, rain],
                                "Unidad": [UNITS[x] for x in FEATURES]})
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="note">Completá los campos y presioná <strong>Analizar mis condiciones</strong> para ver las tres opciones.</div>', unsafe_allow_html=True)

elif page == "Perfiles de condiciones":
    st.markdown('<div class="kicker-green">02 · CLUSTERING</div>', unsafe_allow_html=True)
    st.header("Perfiles de condiciones por cultivo")
    st.markdown(
        '<div class="note"><strong>¿Qué es un perfil?</strong><br>'
        'Un mismo cultivo puede darse en condiciones distintas. Por ejemplo, el arroz puede aparecer en '
        'zonas más lluviosas y en otras más secas. Para reflejar eso, el modelo agrupa las muestras de '
        'cada cultivo en hasta tres <strong>perfiles</strong>: grupos de muestras con suelo y clima parecidos entre sí.<br><br>'
        'Cada perfil muestra el <strong>promedio</strong> de cada variable (nitrógeno, fósforo, potasio, temperatura, '
        'humedad, pH y lluvia) y el <strong>rango</strong> observado, es decir, el valor mínimo y máximo de las muestras del grupo.<br><br>'
        'Los perfiles no están ordenados de mejor a peor: son simplemente diferentes escenarios en los que el cultivo aparece en el dataset.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    crops = sorted(crop_profiles.keys())
    if not crops:
        st.warning("No hay suficientes muestras para crear perfiles. Cada cultivo necesita al menos seis registros.")
    else:
        crop = st.selectbox("Elegí un cultivo para ver sus perfiles", crops, format_func=lambda x: x)
        details = crop_profiles[crop]
        profiles = details["perfiles"]
        st.markdown(f'<div class="note">Para <strong>{crop}</strong> hay {details["total_muestras"]} muestras disponibles y {len(profiles)} perfiles agrupados.</div>', unsafe_allow_html=True)
        st.write("")
        ids = sorted(profiles.keys())
        tabs = st.tabs([f"Perfil {position}" for position in range(1, len(ids) + 1)])
        for position, (tab, pid) in enumerate(zip(tabs, ids), start=1):
            prof = profiles[pid]
            with tab:
                st.markdown(f'<div class="profile"><span class="kicker-green">PERFIL {position}</span><h3>{prof["cantidad"]} muestras</h3><p>Grupo de muestras con condiciones parecidas. Abajo ves el promedio de cada variable y el rango (mínimo – máximo) observado en el grupo.</p></div>', unsafe_allow_html=True)
                metric_cols = st.columns(4, gap="medium")
                for feature_index, feature in enumerate(FEATURES):
                    avg = float(prof["promedios"][feature])
                    low = float(prof["minimos"][feature])
                    high = float(prof["maximos"][feature])
                    with metric_cols[feature_index % 4]:
                        st.markdown(f"**{NAMES[feature]}** ({UNITS[feature]})")
                        st.metric(label="Promedio", value=f"{avg:.2f}")
                        st.caption(f"Rango: {low:.2f} – {high:.2f}")
        st.markdown("### Comparación rápida")
        comparison = pd.DataFrame({
            f"Perfil {i+1}": [round(float(profiles[pid]["promedios"][f]), 2) for f in FEATURES]
            for i, pid in enumerate(ids)
        }, index=[f"{NAMES[f]} ({UNITS[f]})" for f in FEATURES])
        st.dataframe(comparison, use_container_width=True)
        st.markdown('<div class="note"><strong>Interpretación:</strong> los promedios y rangos describen las muestras agrupadas. No deben interpretarse como condiciones óptimas universales sin validación agronómica.</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="kicker-green">TRANSPARENCIA</div>', unsafe_allow_html=True)
    st.header("Sobre los modelos")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown('<div class="panel"><div class="kicker-green">MÉTODO 01</div><h3>Random Forest</h3><p>Combina árboles de decisión para clasificar cultivos a partir de siete variables del suelo y el ambiente.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="kicker-green">MÉTODO 02</div><h3>K-Means</h3><p>Agrupa muestras de cada cultivo según similitudes entre las variables y resume sus promedios y rangos.</p></div>', unsafe_allow_html=True)
    st.write("")
    a, b, c = st.columns(3)
    a.metric("Registros", f"{pack['filas_dataset']:,}")
    b.metric("Cultivos", len(pack["cultivos"]))
    c.metric("Perfiles creados", sum(len(v["perfiles"]) for v in crop_profiles.values()))
    st.metric("Exactitud en el conjunto de prueba", f"{pack['exactitud_prueba'] * 100:.1f}%")
    st.caption("Esta exactitud se calcula en una partición del dataset y no garantiza el mismo desempeño con datos nuevos.")
    with st.expander("Variables utilizadas"):
        st.dataframe(pd.DataFrame({"Variable": [NAMES[f] for f in FEATURES],
                                   "Columna": FEATURES,
                                   "Unidad": [UNITS[f] for f in FEATURES]}),
                     use_container_width=True, hide_index=True)
    with st.expander("Archivos del proyecto"):
        st.markdown("- `merged_crop_dataset.csv`: dataset original.")
        st.markdown("- `crop_model.joblib`: modelo de clasificación guardado.")
        st.markdown("- `crop_clusters.joblib`: perfiles de clustering guardados.")
        st.markdown("Si cambiás el dataset, cerrá la app y eliminá los dos archivos `.joblib` para volver a entrenar.")

st.markdown('<div class="footer">Herramienta educativa basada en datos · Validá las decisiones con información agronómica local.</div>', unsafe_allow_html=True)
