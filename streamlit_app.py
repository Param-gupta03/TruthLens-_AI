import os
import sys
import time
import json
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="TruthLens AI - Fact Checking & Claim Verification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add ml/src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "ml", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .verdict-supports {
        background-color: #ECFDF5;
        color: #065F46;
        border: 2px solid #10B981;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .verdict-refutes {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .verdict-nei {
        background-color: #FFFBEB;
        color: #92400E;
        border: 2px solid #F59E0B;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .evidence-card {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        background-color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading TruthLens AI Pipeline (Model A & Model B)...")
def load_pipeline():
    from fact_check_pipeline import FactCheckPipeline
    return FactCheckPipeline.get_instance()


# Sample test claims
SAMPLE_PRESETS = {
    "Select a preset claim...": {
        "claim": "",
        "evidence": ""
    },
    "✅ True Claim (World Knowledge)": {
        "claim": "Paris is the capital and most populous city of France.",
        "evidence": "Paris is the capital and most populous city of France, with an estimated population of over 2 million residents in the city proper.\nFrance's largest metropolis serves as the country's political, economic, and cultural center."
    },
    "❌ False Claim (Refuted)": {
        "claim": "The Earth is completely flat and stationary at the center of the universe.",
        "evidence": "Satellite imagery, circumnavigation, and astrophysics definitively prove the Earth is an oblate spheroid orbiting the Sun.\nModern observational astronomy confirms that planetary bodies are spherical due to gravitational pull."
    },
    "🔬 Scientific Claim (SciFact)": {
        "claim": "Interleukin-6 expression correlates with disease severity in inflammatory conditions.",
        "evidence": "Elevated serum levels of interleukin-6 (IL-6) were observed in patients with active systemic inflammation and correlated directly with clinical severity markers.\nUnrelated control experiments showed baseline antibody titers in non-inflammatory cohorts."
    },
    "⚠️ Insufficient Information": {
        "claim": "Quantum computers will completely replace all personal laptops by next year.",
        "evidence": "Quantum computing research has achieved significant milestones in superconducting qubits and fault tolerance in enterprise labs."
    }
}

# Sidebar settings
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.title("TruthLens AI")
    st.markdown("**Dual-Model Neural Fact-Checking Architecture**")
    st.markdown("---")

    st.subheader("⚙️ Pipeline Configuration")
    threshold = st.slider(
        "Model A Relevance Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Calibrated threshold to filter candidate passages into relevant evidence."
    )

    preset_choice = st.selectbox(
        "💡 Quick Test Presets",
        options=list(SAMPLE_PRESETS.keys()),
        index=0
    )

    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown("- **Model A**: Evidence Relevance Ranker (`roberta-base`)")
    st.markdown("- **Model B**: Claim Verification Classifier (`roberta-base`)")
    st.markdown("- **Pipeline**: 2-stage verification with domain generalization")


# Main UI Header
st.markdown('<div class="main-header">TruthLens AI - Automated Claim Verification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Verify claims against candidate evidence passages with relevance-weighted reasoning</div>', unsafe_allow_html=True)

# Pre-populate if preset chosen
default_claim = SAMPLE_PRESETS[preset_choice]["claim"] if preset_choice != "Select a preset claim..." else ""
default_evidence = SAMPLE_PRESETS[preset_choice]["evidence"] if preset_choice != "Select a preset claim..." else ""

with st.container():
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown("#### 1. Claim")
        claim_input = st.text_area(
            "Enter the claim to be verified:",
            value=default_claim,
            height=110,
            placeholder="e.g., Paris is the capital city of France."
        )

    with col2:
        st.markdown("#### 2. Candidate Evidence Passages")
        evidence_input = st.text_area(
            "Enter evidence passages (one per line):",
            value=default_evidence,
            height=110,
            placeholder="Passage 1\nPassage 2\nPassage 3"
        )

verify_button = st.button("🚀 Verify Claim", type="primary", use_container_width=True)

if verify_button:
    if not claim_input.strip():
        st.error("Please enter a claim to verify.")
    elif not evidence_input.strip():
        st.error("Please enter at least one evidence passage.")
    else:
        candidate_list = [line.strip() for line in evidence_input.split("\n") if line.strip()]

        try:
            pipeline = load_pipeline()
            start_time = time.time()
            with st.spinner("Analyzing evidence relevance & verifying claim veracity..."):
                result = pipeline.run_fact_check(
                    claim=claim_input.strip(),
                    candidate_evidence=candidate_list,
                    custom_threshold=threshold
                )
            latency = round((time.time() - start_time) * 1000, 1)

            verif = result.get("verification", {})
            label = verif.get("label", "NOT_ENOUGH_INFO")
            confidence = verif.get("confidence", 0.0)
            probs = verif.get("probabilities", {})

            st.markdown("---")
            st.markdown("### 📊 Verification Verdict")

            # Verdict banner
            if label == "SUPPORTS":
                st.markdown(f'<div class="verdict-supports">VERDICT: SUPPORTS ({confidence * 100:.1f}% Confidence)</div>', unsafe_allow_html=True)
            elif label == "REFUTES":
                st.markdown(f'<div class="verdict-refutes">VERDICT: REFUTES ({confidence * 100:.1f}% Confidence)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-nei">VERDICT: NOT ENOUGH INFO ({confidence * 100:.1f}% Confidence)</div>', unsafe_allow_html=True)

            # Key metrics
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Final Verdict", label)
            m2.metric("Verdict Confidence", f"{confidence * 100:.1f}%")
            m3.metric("Relevant Evidence", f"{result.get('relevant_evidence_count', 0)} / {len(candidate_list)}")
            m4.metric("Inference Latency", f"{latency} ms")

            # Probability Breakdown
            st.markdown("#### Probability Distribution")
            p_cols = st.columns(3)
            with p_cols[0]:
                sup_p = float(probs.get("SUPPORTS", 0.0))
                st.write(f"**SUPPORTS**: {sup_p * 100:.1f}%")
                st.progress(min(max(sup_p, 0.0), 1.0))
            with p_cols[1]:
                ref_p = float(probs.get("REFUTES", 0.0))
                st.write(f"**REFUTES**: {ref_p * 100:.1f}%")
                st.progress(min(max(ref_p, 0.0), 1.0))
            with p_cols[2]:
                nei_p = float(probs.get("NOT_ENOUGH_INFO", 0.0))
                st.write(f"**NOT_ENOUGH_INFO**: {nei_p * 100:.1f}%")
                st.progress(min(max(nei_p, 0.0), 1.0))

            # Evidence Breakdown
            st.markdown("---")
            st.markdown("#### 📑 Evidence Passages & Relevance Filtering")
            evidence_results = result.get("evidence_results", [])

            for idx, ev_item in enumerate(evidence_results, start=1):
                ev_text = ev_item.get("evidence", "")
                rel_label = ev_item.get("relevance_label", "NOT_RELEVANT")
                rel_score = ev_item.get("relevance_score", 0.0)

                badge = "🟢 RELEVANT" if rel_label == "RELEVANT" else "⚪ NOT RELEVANT"
                with st.expander(f"Passage #{idx}: {badge} (Score: {rel_score:.3f})", expanded=(rel_label == "RELEVANT")):
                    st.write(ev_text)
                    st.caption(f"Relevance Score: **{rel_score:.4f}** (Threshold: {threshold})")

            # JSON inspect
            with st.expander("🛠️ View Full Raw Prediction JSON"):
                st.json(result)

        except Exception as e:
            st.error(f"Error during verification: {str(e)}")
            st.exception(e)
