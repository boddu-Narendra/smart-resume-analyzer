import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import re
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import string
import nltk

try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# ---------------------------
# Load pretrained model
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ---------------------------
# Skill extraction function
# ---------------------------
def extract_skills(text):
    skills = [
        "Python", "Java", "C++", "SQL", "HTML", "CSS", "JavaScript", "React", "Node.js",
        "Machine Learning", "Deep Learning", "NLP", "Data Science", "TensorFlow", "PyTorch",
        "Excel", "Communication", "Leadership", "Project Management"
    ]
    found = []
    for skill in skills:
        patt = rf"\b{re.escape(skill)}\b"
        if re.search(patt, text, re.IGNORECASE):
            found.append(skill)
    return list(set(found))

# ---------------------------
# Learning resources mapping
# ---------------------------
learning_resources = {
    "Python": "https://www.w3schools.com/python/",
    "SQL": "https://www.sqlbolt.com/",
    "Machine Learning": "https://www.coursera.org/learn/machine-learning",
    "Deep Learning": "https://www.deeplearning.ai/",
    "TensorFlow": "https://www.tensorflow.org/tutorials",
    "PyTorch": "https://pytorch.org/tutorials/",
    "React": "https://react.dev/learn",
    "Project Management": "https://www.pmi.org/certifications/capm",
    "Java": "https://www.w3schools.com/java/",
    "C++": "https://www.learncpp.com/",
    "HTML": "https://www.w3schools.com/html/",
    "CSS": "https://www.w3schools.com/css/",
    "JavaScript": "https://www.javascript.com/"
}

# ---------------------------
# Motivational Quotes
# ---------------------------
quotes = [
    "🌟 Believe in yourself, recruiters love confidence!",
    "🚀 Every skill you add makes you unstoppable.",
    "🔥 Great resumes are written, not born — keep improving!",
    "💡 Small improvements lead to big opportunities.",
    "🏆 You're closer to your dream job than you think!"
]

# ---------------------------
# Job Suggestions
# ---------------------------
job_suggestions = {
    "Python": ["Data Analyst", "Machine Learning Engineer", "Backend Developer"],
    "Java": ["Software Engineer", "Spring Boot Developer", "Android Developer"],
    "SQL": ["Database Administrator", "Data Engineer", "Business Analyst"],
    "React": ["Frontend Developer", "Full Stack Developer", "UI Engineer"],
    "Machine Learning": ["AI Engineer", "Data Scientist", "ML Researcher"],
    "Deep Learning": ["Computer Vision Engineer", "AI Scientist", "NLP Engineer"],
    "Project Management": ["Project Manager", "Scrum Master", "Agile Coach"]
}

# ---------------------------
# Streamlit UI Setup
# ---------------------------
st.set_page_config(page_title="Smart Resume Analyzer", page_icon="📄", layout="wide")

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f9f9f9, #dbeafe);
}
h1, h2, h3, h4 {
    color: #1e3a8a;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["Upload Resume", "Analysis"])

# ---------------------------
# Upload Page
# ---------------------------
if page == "Upload Resume":
    st.title("📂 Upload & Job Description")
    resume_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])
    job_description = st.text_area("📝 Paste Job Description here")

    if resume_file and job_description:
        st.session_state["resume_file"] = resume_file
        st.session_state["job_description"] = job_description
        st.success("✅ File & Job Description uploaded! Go to 'Analysis' page to see results.")

# ---------------------------
# Analysis Page
# ---------------------------
if page == "Analysis":
    if "resume_file" not in st.session_state or "job_description" not in st.session_state:
        st.warning("⚠ Please upload a resume and job description first.")
    else:
        resume_file = st.session_state["resume_file"]
        job_description = st.session_state["job_description"]

        # Extract resume text
        resume_text = ""
        with pdfplumber.open(resume_file) as pdf:
            for page_ in pdf.pages:
                resume_text += page_.extract_text() or ""

        # Encode for similarity
        resume_emb = model.encode(resume_text, convert_to_tensor=True)
        jd_emb = model.encode(job_description, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(resume_emb, jd_emb).item()

        # Skill analysis
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(job_description)
        missing_skills = list(set(jd_skills) - set(resume_skills))

        # ---------------------------
        # Match Percentage Chart
        # ---------------------------
        st.subheader("📊 Match Overview")

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.pie([similarity, 1 - similarity],
               startangle=90,
               colors=["#10b981", "#e5e7eb"],
               wedgeprops=dict(width=0.3))
        ax.text(0, 0, f"{similarity*100:.1f}%", ha='center', va='center',
                fontsize=12, fontweight='bold', color="#1e3a8a")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(fig, clear_figure=True)

        if similarity > 0.75:
            st.success("🔥 Excellent Match! Your resume aligns strongly with the job description.")
            st.balloons()
        elif similarity > 0.5:
            st.warning("⚡ Moderate Match. Some improvements needed to stand out.")
        else:
            st.error("❌ Weak Match. Your resume needs more alignment with the job description.")

        st.info(random.choice(quotes))

        stop_words = set(stopwords.words('english'))

        def preprocess(text):
            words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
            return set([w for w in words if w not in stop_words])

        # ---------------------------
        # ATS Simulation
        # ---------------------------
        st.subheader("📌 ATS Simulation")
        jd_keywords = preprocess(job_description)
        resume_keywords = preprocess(resume_text)
        overlap = len(jd_keywords & resume_keywords)
        ats_score = (overlap / len(jd_keywords)) * 100 if jd_keywords else 0

        # ✅ New: Display the ATS Score
        st.metric(label="✅ ATS Score", value=f"{ats_score:.2f}%")

        if ats_score > 75:
            st.success("✅ Excellent Keyword Match! Your resume is highly optimized for ATS.")
        elif ats_score > 50:
            st.warning("⚠ Fair ATS Match. Consider adding more relevant keywords.")
        else:
            st.error("❌ Poor ATS Match. Try including more job-relevant keywords in your resume.")

        # ---------------------------
        # Resume Grade
        # ---------------------------
        st.subheader("🎓 Resume Grade")
        if similarity > 0.8 and len(missing_skills) < 2:
            grade = "A"
        elif similarity > 0.6:
            grade = "B"
        elif similarity > 0.4:
            grade = "C"
        else:
            grade = "D"
        st.success(f"Your Resume Grade: **{grade}**")

        # 🧠 Skill Display
        st.subheader("🧠 Skill Analysis")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("### ✅ Resume Skills")
            if resume_skills:
                st.markdown('<div style="background:#f0f9ff;padding:10px;border-radius:10px;border:1px solid black;">',
                            unsafe_allow_html=True)
                for skill in sorted(resume_skills):
                    st.markdown(f"• {skill}")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.write("❌ No skills found")

        with c2:
            st.markdown("### 📌 JD Skills")
            if jd_skills:
                st.markdown('<div style="background:#fff7e6;padding:10px;border-radius:10px;border:1px solid black;">',
                            unsafe_allow_html=True)
                for skill in sorted(jd_skills):
                    st.markdown(f"• {skill}")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.write("❌ No skills found")

        with c3:
            st.markdown("### ⚠ Missing Skills")
            if missing_skills:
                st.markdown('<div style="background:#ffe6e6;padding:10px;border-radius:10px;border:1px solid black;">',
                            unsafe_allow_html=True)
                for skill in sorted(missing_skills):
                    st.markdown(f"• {skill}")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.success("✅ No major skills missing!")

       

        # ---------------------------
        # Resume Improvement Suggestions
        # ---------------------------
        st.subheader("💡 Resume Improvement Suggestions")
        if grade in ["C", "D"]:
            st.write("• Add measurable achievements (e.g., 'Improved accuracy by 20%').")
            st.write("• Align your skills section with job description keywords.")
            st.write("• Consider shortening long paragraphs into bullet points.")
        else:
            st.write("✅ Your resume structure looks strong. Focus on polishing details.")

        # ---------------------------
        # Job Suggestions
        # ---------------------------
        st.subheader("💼 Suggested Jobs for Your Skills")
        suggested_jobs = []
        for skill in resume_skills:
            if skill in job_suggestions:
                suggested_jobs.extend(job_suggestions[skill])
        suggested_jobs = sorted(list(set(suggested_jobs)))

        if suggested_jobs:
            st.markdown('<div style="background:#e0f7fa;padding:10px;border-radius:10px;border:1px solid black;">',
                        unsafe_allow_html=True)
            for job in suggested_jobs:
                st.markdown(f"• {job}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.write("No job suggestions available based on detected skills.")

        # ---------------------------
        # Career Path Predictor
        # ---------------------------
        st.subheader("🔮 Career Path Predictor")
        if resume_skills:
            if "Python" in resume_skills and "Machine Learning" in resume_skills:
                st.write("• Data Scientist → Senior Data Scientist → AI Architect")
            if "JavaScript" in resume_skills and "React" in resume_skills:
                st.write("• Frontend Developer → Full Stack Developer → Tech Lead")
            if "Java" in resume_skills:
                st.write("• Java Developer → Backend Specialist → Solution Architect")
        else:
            st.write("Add more skills to see a clearer career path.")

        # ---------------------------
        # Job Search Platforms
        # ---------------------------
        if suggested_jobs:
            st.subheader("🌐 Explore Jobs on Platforms")
            st.markdown('<div style="background:#f3f4f6;padding:10px;border-radius:10px;border:1px solid black;">',
                        unsafe_allow_html=True)
            for job in suggested_jobs:
                query = job.replace(" ", "+")
                st.markdown(
                    f"🔎 **{job}** → "
                    f"[LinkedIn](https://www.linkedin.com/jobs/search/?keywords={query}) | "
                    f"[Naukri](https://www.naukri.com/{query}-jobs) | "
                    f"[Indeed](https://www.indeed.com/jobs?q={query}) | "
                    f"[Glassdoor](https://www.glassdoor.com/Job/jobs.htm?sc.keyword={query})"
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # =====================================================
        # 🚀 New Advanced Features
        # =====================================================
        st.subheader("📌 Personalized Career Roadmap")
        if missing_skills:
            st.markdown("Here’s a suggested roadmap to boost your career profile:")
            for i, skill in enumerate(missing_skills, 1):
                st.markdown(f"**Step {i}:** Learn {skill} → {learning_resources.get(skill,'Search Online')}")
        else:
            st.success("🎉 You already have most required skills, focus on advanced projects!")

        st.subheader("📈 Skill Evolution Timeline (Next 6 Months)")
        months = ["Month 1","Month 2","Month 3","Month 4","Month 5","Month 6"]
        skill_growth = np.cumsum(np.random.randint(5,15,len(months)))
        fig2, ax2 = plt.subplots()
        ax2.plot(months, skill_growth, marker="o", color="#1e3a8a")
        ax2.set_ylabel("Skill Proficiency %")
        ax2.set_ylim(0,100)
        ax2.set_title("Projected Skill Evolution")
        st.pyplot(fig2, clear_figure=True)

        st.subheader("🎯 Hiring Probability Score")
        hiring_score = (similarity*70) + (100-len(missing_skills)*5)
        hiring_score = max(0, min(100, hiring_score))
        st.progress(int(hiring_score))
        st.success(f"Estimated Hiring Probability: **{hiring_score:.1f}%**")

        # ---------------------------
        # Salary Insights
        # ---------------------------
        st.subheader("💰 Salary Insights (Estimated)")
        if "Software Developer" in suggested_jobs:
            st.write("📊 Software Developer: ₹5–12 LPA in India, $70K–110K in US")

        if "Web Developer" in suggested_jobs:
            st.write("📊 Web Developer: ₹4–10 LPA in India, $65K–100K in US")

        if "Mobile App Developer" in suggested_jobs:
            st.write("📊 Mobile App Developer: ₹5–12 LPA in India, $70K–115K in US")

        if "Database Administrator" in suggested_jobs:
            st.write("📊 Database Administrator: ₹6–14 LPA in India, $75K–115K in US")

        if "Cloud Engineer" in suggested_jobs:
            st.write("📊 Cloud Engineer: ₹7–18 LPA in India, $95K–140K in US")

        if "Network Engineer" in suggested_jobs:
            st.write("📊 Network Engineer: ₹4–9 LPA in India, $65K–95K in US")

        if "Cybersecurity Analyst" in suggested_jobs:
            st.write("📊 Cybersecurity Analyst: ₹6–15 LPA in India, $80K–120K in US")

        if "AI Engineer" in suggested_jobs:
            st.write("📊 AI Engineer: ₹10–22 LPA in India, $110K–160K in US")

        if "Machine Learning Engineer" in suggested_jobs:
            st.write("📊 Machine Learning Engineer: ₹9–20 LPA in India, $105K–150K in US")

        if "Data Scientist" in suggested_jobs:
            st.write("📊 Data Scientist: ₹8–18 LPA in India, $90K–130K in US")

        if "Data Analyst" in suggested_jobs:
            st.write("📊 Data Analyst: ₹5–12 LPA in India, $65K–95K in US")

        if "DevOps Engineer" in suggested_jobs:
            st.write("📊 DevOps Engineer: ₹7–16 LPA in India, $95K–135K in US")

        if "IT Support Specialist" in suggested_jobs:
            st.write("📊 IT Support Specialist: ₹3–8 LPA in India, $50K–80K in US")

        if "System Administrator" in suggested_jobs:
            st.write("📊 System Administrator: ₹4–10 LPA in India, $60K–90K in US")

        if "ERP Consultant" in suggested_jobs:
            st.write("📊 ERP Consultant: ₹8–20 LPA in India, $95K–140K in US")

        if "CRM Specialist" in suggested_jobs:
            st.write("📊 CRM Specialist: ₹6–14 LPA in India, $80K–120K in US")

        if "Blockchain Developer" in suggested_jobs:
            st.write("📊 Blockchain Developer: ₹8–20 LPA in India, $100K–150K in US")

        if "IoT Engineer" in suggested_jobs:
            st.write("📊 IoT Engineer: ₹6–15 LPA in India, $85K–125K in US")

        if "AR/VR Developer" in suggested_jobs:
            st.write("📊 AR/VR Developer: ₹7–16 LPA in India, $90K–135K in US")

        if "Project Manager" in suggested_jobs:
            st.write("📊 Project Manager: ₹10–25 LPA in India, $95K–140K in US")
