import streamlit as st
import pdfplumber
import whisper
import os
import re
import tempfile
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# -------------------- Helper Functions --------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def normalize(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def extract_pdf_text(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

def transcribe_audio(uploaded_audio):
    suffix = os.path.splitext(uploaded_audio.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_audio.read())
        tmp_path = tmp_file.name
    model = load_whisper_model()
    result = model.transcribe(tmp_path)
    os.remove(tmp_path)
    return result["text"]

# -------------------- UI Components --------------------

st.title("🎯 TalentMatch AI")

# --- Upload JD ---
st.subheader("📝 Upload Job Description (JD)")
jd_file = st.file_uploader("Upload JD (PDF)", type="pdf")
jd_text = ""

if jd_file:
    jd_text = extract_pdf_text(jd_file)
    st.text_area("Extracted JD Text", jd_text, height=200)

# --- Audio Option ---
include_audio = st.radio("Include audio for each candidate?", ["No", "Yes"]) == "Yes"

# --- Number of Resumes ---
num_resumes = st.number_input("How many resumes do you want to evaluate?", min_value=1, max_value=20, step=1)

# --- Resume & Audio Upload ---
st.subheader("📂 Upload Candidate Files")
resume_files = []
audio_files = []
candidate_names = []

for i in range(num_resumes):
    st.markdown(f"#### Candidate #{i+1}")
    name = st.text_input(f"Candidate #{i+1} Name", value=f"Candidate #{i+1}", key=f"name_{i}")
    resume = st.file_uploader("Upload Resume (PDF)", type="pdf", key=f"resume_{i}")
    resume_files.append(resume)
    candidate_names.append(name)

    if include_audio:
        audio = st.file_uploader("Upload Audio (MP3/WAV/M4A)", type=["mp3", "wav", "m4a"], key=f"audio_{i}")
        audio_files.append(audio)
    else:
        audio_files.append(None)

# -------------------- Main Evaluation --------------------

if jd_file and any(resume_files):
    with st.spinner("🔍 Analyzing candidates..."):
        model = load_embedding_model()
        jd_embedding = model.encode(normalize(jd_text), convert_to_tensor=True)
        results = []

        for i in range(num_resumes):
            resume = resume_files[i]
            audio = audio_files[i]
            name = candidate_names[i]

            if not resume:
                continue

            with st.spinner(f"Processing {name}..."):
                resume_text = extract_pdf_text(resume)
                combined_text = resume_text

                if audio:
                    try:
                        audio_text = transcribe_audio(audio)
                        combined_text += " " + audio_text
                    except Exception as e:
                        st.error(f"Error processing audio for {name}: {str(e)}")

                candidate_embedding = model.encode(normalize(combined_text), convert_to_tensor=True)
                score = util.pytorch_cos_sim(jd_embedding, candidate_embedding).item() * 100
                results.append((name, score))

    # --- Display Results ---
    if results:
        st.subheader("🏆 Top Candidates")
        results.sort(key=lambda x: x[1], reverse=True)

        # Compute average score
        scores = [score for _, score in results]
        average_score = sum(scores) / len(scores)

        # Add Status column
        updated_results = []
        for name, score in results:
            status = "Selected" if score >= average_score else "Not Selected"
            updated_results.append((name, score, status))

        df = pd.DataFrame(updated_results, columns=["Candidate", "Match %", "Status"])
        st.table(df.style.format({"Match %": "{:.2f}"}))

        st.download_button(
            "⬇️ Download Results as CSV",
            df.to_csv(index=False),
            file_name="talentmatch_results.csv",
            mime="text/csv"
        )

        st.info(f"📊 Average Score: **{average_score:.2f}%**")
    else:
        st.warning("No valid resumes found.")
