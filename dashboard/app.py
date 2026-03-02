"""
Healthcare Claims Fraud Detection Dashboard
- User authentication with role-based access
- Real-time fraud detection monitoring
- User management for superusers
Now using SQLAlchemy ORM — no raw psycopg2.
"""
import os
import sys
import uuid
import requests
from datetime import datetime

import bcrypt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import func, cast, Date
from sqlalchemy.exc import IntegrityError

# ── SQLAlchemy shared layer ────────────────────────────────────────────────────
sys.path.insert(0, '/app')
from shared.database import get_session
from shared.models import User, Claim, FraudAnalysis

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Auth helpers ───────────────────────────────────────────────────────────────
def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def authenticate_user(username: str, password: str):
    """Return user dict on success, else None."""
    try:
        with get_session() as session:
            user = (
                session.query(User)
                .filter(User.username == username, User.is_active == True)
                .first()
            )
            if user and verify_password(password, user.password_hash):
                user.last_login = datetime.utcnow()
                session.flush()
                # Detach from session so we can return a plain dict
                return {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "role": user.role,
                    "is_active": user.is_active,
                }
        return None
    except Exception as e:
        st.error(f"Database error during authentication: {e}")
        return None


def is_default_admin_password() -> bool:
    DEFAULT_HASH = "$2b$12$DvfarmHo0abG3iqd1kjLDewOkhWW0Ldgu333J.U04/IoyO6wrFVNi"
    try:
        with get_session() as session:
            row = session.query(User.password_hash).filter(User.username == "admin").first()
            return row is not None and row[0] == DEFAULT_HASH
    except Exception:
        return False


def change_password(user_id: int, old_password: str, new_password: str):
    try:
        with get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return False, "User not found"
            if not verify_password(old_password, user.password_hash):
                return False, "Current password is incorrect"
            user.password_hash = hash_password(new_password)
        return True, "Password changed successfully"
    except Exception as e:
        return False, f"Database error: {e}"


# ── Data helpers ───────────────────────────────────────────────────────────────
def get_fraud_stats() -> dict:
    try:
        with get_session() as session:
            total_claims    = session.query(func.count(Claim.id)).scalar() or 0
            analyzed_claims = session.query(func.count(FraudAnalysis.id)).scalar() or 0
            anomalies       = session.query(func.count(FraudAnalysis.id)).filter(
                                  FraudAnalysis.is_anomaly == True).scalar() or 0

            risk_rows = (
                session.query(FraudAnalysis.risk_level, func.count(FraudAnalysis.id))
                .group_by(FraudAnalysis.risk_level)
                .all()
            )
            risk_distribution = {row[0]: row[1] for row in risk_rows}

        return {
            "total_claims":    total_claims,
            "analyzed_claims": analyzed_claims,
            "anomalies":       anomalies,
            "fraud_rate":      (anomalies / analyzed_claims * 100) if analyzed_claims > 0 else 0,
            "risk_distribution": risk_distribution,
        }
    except Exception as e:
        st.error(f"Error fetching statistics: {e}")
        return {"total_claims": 0, "analyzed_claims": 0, "anomalies": 0,
                "fraud_rate": 0, "risk_distribution": {}}


def get_recent_claims(limit: int = 50, risk_filter: str = None) -> pd.DataFrame:
    try:
        with get_session() as session:
            q = (
                session.query(
                    Claim.claim_id, Claim.beneficiary_id, Claim.provider_id,
                    Claim.claim_amount, Claim.age, Claim.claim_type, Claim.created_at,
                    FraudAnalysis.is_anomaly, FraudAnalysis.anomaly_score,
                    FraudAnalysis.risk_level, FraudAnalysis.analyzed_at,
                    FraudAnalysis.reviewed,
                )
                .outerjoin(FraudAnalysis, Claim.claim_id == FraudAnalysis.claim_id)
            )
            if risk_filter and risk_filter != "All":
                q = q.filter(FraudAnalysis.risk_level == risk_filter)
            q = q.order_by(Claim.created_at.desc()).limit(limit)
            rows = q.all()

        if rows:
            cols = [
                "claim_id", "beneficiary_id", "provider_id", "claim_amount",
                "age", "claim_type", "created_at", "is_anomaly", "anomaly_score",
                "risk_level", "analyzed_at", "reviewed",
            ]
            return pd.DataFrame(rows, columns=cols)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching claims: {e}")
        return pd.DataFrame()


# ── UI components ──────────────────────────────────────────────────────────────
def login_page():
    st.title("🔐 Healthcare Fraud Detection System")
    st.markdown("### Please log in to continue")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit   = st.form_submit_button("Login", use_container_width=True)

            if submit:
                if username and password:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state["user"]          = user
                        st.session_state["authenticated"] = True
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")

        if is_default_admin_password():
            st.info("**Default credentials:** username: `admin`, password: `admin123`")
            st.warning("⚠️ Please change the default password after first login!")


def logout():
    st.session_state["authenticated"] = False
    st.session_state["user"]          = None
    st.rerun()


def show_change_password_sidebar():
    with st.sidebar.expander("🔑 Change Password"):
        with st.form("change_pw_form"):
            old_pw     = st.text_input("Current Password",    type="password", key="old_pw")
            new_pw     = st.text_input("New Password",        type="password", key="new_pw")
            confirm_pw = st.text_input("Confirm New Password",type="password", key="confirm_pw")
            submitted  = st.form_submit_button("Update Password")
            if submitted:
                if not old_pw or not new_pw or not confirm_pw:
                    st.warning("Please fill in all fields")
                elif new_pw != confirm_pw:
                    st.error("New passwords do not match")
                elif len(new_pw) < 8:
                    st.error("Password must be at least 8 characters")
                else:
                    ok, msg = change_password(
                        st.session_state["user"]["id"], old_pw, new_pw
                    )
                    st.success(msg) if ok else st.error(msg)


def main_dashboard():
    st.title("🔍 Healthcare Claims Fraud Detection Dashboard")

    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state['user']['username']}!")
        st.markdown(f"**Role:** {st.session_state['user']['role'].upper()}")
        st.markdown("---")

        base_pages = ["📊 Dashboard", "📋 Claims List", "📝 Submit Claim", "📈 Analytics"]
        nav_pages  = (base_pages + ["👥 User Management"]
                      if st.session_state["user"]["role"] == "superuser"
                      else base_pages)

        page = st.radio("Navigation", nav_pages)
        st.markdown("---")
        show_change_password_sidebar()
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            logout()

        st.markdown("---")
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        if auto_refresh:
            import time
            time.sleep(30)
            st.rerun()

    if   page == "📊 Dashboard":        show_dashboard()
    elif page == "📋 Claims List":      show_claims_list()
    elif page == "📝 Submit Claim":     show_submit_claim()
    elif page == "📈 Analytics":        show_analytics()
    elif page == "👥 User Management":  show_user_management()


# ── Dashboard page ─────────────────────────────────────────────────────────────
def show_dashboard():
    st.header("Overview")
    stats = get_fraud_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Claims",       f"{stats['total_claims']:,}")
    col2.metric("Analyzed",           f"{stats['analyzed_claims']:,}")
    col3.metric("Anomalies Detected", f"{stats['anomalies']:,}")
    col4.metric("Fraud Rate",         f"{stats['fraud_rate']:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Level Distribution")
        risk_dist = stats["risk_distribution"]
        if risk_dist:
            fig = px.pie(
                values=list(risk_dist.values()),
                names=list(risk_dist.keys()),
                color=list(risk_dist.keys()),
                color_discrete_map={
                    "CRITICAL": "#FF0000", "HIGH": "#FF6B6B",
                    "MEDIUM": "#FFA500",   "LOW":  "#4CAF50",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available yet")

    with col2:
        st.subheader("Recent High-Risk Claims")
        df = get_recent_claims(limit=10, risk_filter="HIGH")
        if not df.empty:
            df_d = df[["claim_id", "claim_amount", "anomaly_score", "risk_level"]].copy()
            df_d["anomaly_score"] = df_d["anomaly_score"].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            df_d["claim_amount"]  = df_d["claim_amount"].apply(
                lambda x: f"${float(x):,.2f}" if pd.notna(x) else "N/A")
            st.dataframe(df_d, use_container_width=True, hide_index=True)
        else:
            st.info("No high-risk claims detected")


# ── Claims list page ───────────────────────────────────────────────────────────
def show_claims_list():
    st.header("Claims List")

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.selectbox("Filter by Risk Level",
                                   ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with col2:
        limit = st.number_input("Number of claims", min_value=10, max_value=500, value=50)
    with col3:
        st.markdown(" ")
        st.markdown(" ")
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    df = get_recent_claims(limit=limit,
                           risk_filter=risk_filter if risk_filter != "All" else None)

    if not df.empty:
        df_display = df[[
            "claim_id", "beneficiary_id", "provider_id", "claim_amount",
            "age", "claim_type", "risk_level", "anomaly_score", "reviewed", "created_at",
        ]].copy()

        df_display["claim_amount"]  = df_display["claim_amount"].apply(
            lambda x: f"${float(x):,.2f}")
        df_display["anomaly_score"] = df_display["anomaly_score"].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        df_display["reviewed"]      = df_display["reviewed"].apply(lambda x: "✓" if x else "")

        def highlight_risk(row):
            if row["risk_level"] == "CRITICAL":
                return ["background-color: #ffebee"] * len(row)
            if row["risk_level"] == "HIGH":
                return ["background-color: #fff3e0"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_display.style.apply(highlight_risk, axis=1),
            use_container_width=True, hide_index=True, height=600,
        )
        st.info(f"Showing {len(df)} claims")
    else:
        st.info("No claims found with the selected filters")


# ── User management page ───────────────────────────────────────────────────────
def show_user_management():
    if st.session_state["user"]["role"] != "superuser":
        st.error("Access denied. Superuser privileges required.")
        return

    st.header("👥 User Management")
    tab1, tab2 = st.tabs(["View Users", "Add New User"])

    with tab1:
        try:
            with get_session() as session:
                users = (
                    session.query(
                        User.id, User.username, User.email, User.role,
                        User.is_active, User.created_at, User.last_login,
                    )
                    .order_by(User.created_at.desc())
                    .all()
                )
            if users:
                cols = ["id", "username", "email", "role",
                        "is_active", "created_at", "last_login"]
                df = pd.DataFrame(users, columns=cols)
                df["is_active"] = df["is_active"].apply(lambda x: "✓" if x else "✗")
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No users found")
        except Exception as e:
            st.error(f"Error fetching users: {e}")

    with tab2:
        with st.form("add_user_form"):
            st.subheader("Create New User")
            new_username = st.text_input("Username")
            new_email    = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            new_role     = st.selectbox("Role", ["viewer", "analyst", "superuser"])
            submit       = st.form_submit_button("Create User")

            if submit:
                if new_username and new_email and new_password:
                    try:
                        with get_session() as session:
                            new_user = User(
                                username      = new_username,
                                email         = new_email,
                                password_hash = hash_password(new_password),
                                role          = new_role,
                                created_by    = st.session_state["user"]["id"],
                            )
                            session.add(new_user)
                        st.success(f"User '{new_username}' created successfully!")
                        st.rerun()
                    except IntegrityError:
                        st.error("Username or email already exists")
                    except Exception as e:
                        st.error(f"Error creating user: {e}")
                else:
                    st.warning("Please fill in all fields")


# ── Submit claim page ──────────────────────────────────────────────────────────
def show_submit_claim():
    st.header("📝 Submit a Claim")
    st.markdown("Fill in the claim details below. Required fields are marked with *")

    INGESTION_URL = os.getenv("INGESTION_API_URL", "http://ingestion_api:8080/ingest")

    tab_basic, tab_diag, tab_proc, tab_patient, tab_chronic = st.tabs(
        ["Basic Info", "Diagnosis Codes", "Procedure & Physicians",
         "Patient Demographics", "Chronic Conditions"]
    )

    with tab_basic:
        col1, col2 = st.columns(2)
        with col1:
            claim_id       = st.text_input("Claim ID *",
                                           value=f"CLM-{uuid.uuid4().hex[:8].upper()}")
            beneficiary_id = st.text_input("Beneficiary ID (user_id) *")
            provider_id    = st.text_input("Provider ID *")
        with col2:
            claim_type          = st.selectbox("Claim Type", ["OUTPATIENT", "INPATIENT"])
            claim_amount        = st.number_input("Claim Amount ($) *",
                                                  min_value=0.01, max_value=999999.99,
                                                  value=500.00, step=0.01)
            deductible_amt_paid = st.number_input("Deductible Paid ($)",
                                                  min_value=0.0, max_value=999999.99,
                                                  value=0.0, step=0.01)
        col3, col4 = st.columns(2)
        with col3:
            claim_start_date = st.date_input("Claim Start Date", value=None)
        with col4:
            claim_end_date   = st.date_input("Claim End Date",   value=None)

    with tab_diag:
        st.markdown("**Primary diagnosis code is required. Secondary codes are optional.**")
        primary_diagnosis_code = st.text_input("Primary ICD Code (icd_code) *",
                                               placeholder="e.g. 4019")
        admit_diagnosis_code   = st.text_input("Admit Diagnosis Code", placeholder="e.g. 4019")
        st.markdown("Secondary Diagnosis Codes")
        diag_cols  = st.columns(5)
        diag_codes = []
        for i, col in enumerate(diag_cols * 2):
            if i >= 9:
                break
            with col:
                diag_codes.append(st.text_input(f"Code {i+2}", key=f"diag_{i+2}",
                                                placeholder="optional"))

    with tab_proc:
        st.markdown("**Procedure codes and attending physicians (all optional)**")
        proc_cols  = st.columns(3)
        proc_codes = []
        for i, col in enumerate(proc_cols * 2):
            if i >= 6:
                break
            with col:
                proc_codes.append(st.text_input(f"Procedure Code {i+1}", key=f"proc_{i+1}",
                                                placeholder="optional"))
        col1, col2, col3 = st.columns(3)
        with col1:
            attending_physician = st.text_input("Attending Physician",  placeholder="optional")
        with col2:
            operating_physician = st.text_input("Operating Physician",  placeholder="optional")
        with col3:
            other_physician     = st.text_input("Other Physician",       placeholder="optional")

    with tab_patient:
        col1, col2 = st.columns(2)
        with col1:
            age    = st.number_input("Age *", min_value=0, max_value=120, value=65)
            gender = st.selectbox("Gender", ["Not specified", "Male (1)", "Female (2)"])
            race   = st.selectbox("Race",   ["Not specified", "1", "2", "3", "4", "5"])
        with col2:
            state  = st.number_input("State Code",  min_value=0, max_value=99,  value=0)
            county = st.number_input("County Code", min_value=0, max_value=999, value=0)
            part_a = st.slider("Months Part A Coverage", 0, 12, 12)
            part_b = st.slider("Months Part B Coverage", 0, 12, 12)

    with tab_chronic:
        st.markdown("Select all conditions that apply (checked = Yes)")
        conditions = {
            "chronic_cond_alzheimer":           "Alzheimer's",
            "chronic_cond_heartfailure":        "Heart Failure",
            "chronic_cond_kidneydisease":       "Kidney Disease",
            "chronic_cond_cancer":              "Cancer",
            "chronic_cond_obstrpulmonary":      "Obstructive Pulmonary",
            "chronic_cond_depression":          "Depression",
            "chronic_cond_diabetes":            "Diabetes",
            "chronic_cond_ischemicheart":       "Ischemic Heart",
            "chronic_cond_osteoporasis":        "Osteoporosis",
            "chronic_cond_rheumatoidarthritis": "Rheumatoid Arthritis",
            "chronic_cond_stroke":              "Stroke",
        }
        chronic_values = {}
        cond_cols = st.columns(3)
        for idx, (field, label) in enumerate(conditions.items()):
            with cond_cols[idx % 3]:
                checked = st.checkbox(label, key=f"chk_{field}")
                chronic_values[field] = 1 if checked else 2
        renal = st.selectbox("Renal Disease Indicator", ["0", "Y"])

    st.markdown("---")
    if st.button("🚀 Submit Claim", type="primary", use_container_width=True):
        errors = []
        if not claim_id.strip():             errors.append("Claim ID is required")
        if not beneficiary_id.strip():       errors.append("Beneficiary ID is required")
        if not provider_id.strip():          errors.append("Provider ID is required")
        if not primary_diagnosis_code.strip():errors.append("Primary ICD Code is required")
        if claim_amount <= 0:                errors.append("Claim amount must be > 0")

        if errors:
            for e in errors:
                st.error(e)
        else:
            payload = {
                "claim_id":              claim_id.strip(),
                "user_id":               beneficiary_id.strip(),
                "provider_id":           provider_id.strip(),
                "claim_type":            claim_type,
                "amount":                float(claim_amount),
                "deductible_amt_paid":   float(deductible_amt_paid),
                "icd_code":              primary_diagnosis_code.strip(),
                "admit_diagnosis_code":  admit_diagnosis_code.strip() or None,
                "age":                   int(age),
                "gender":                {"Not specified": None, "Male (1)": 1, "Female (2)": 2}[gender],
                "race":                  None if race == "Not specified" else int(race),
                "state":                 int(state) if state else None,
                "county":                int(county) if county else None,
                "no_of_months_part_a_cov": int(part_a),
                "no_of_months_part_b_cov": int(part_b),
                "renal_disease_indicator": renal,
                "claim_start_date":      str(claim_start_date) if claim_start_date else None,
                "claim_end_date":        str(claim_end_date)   if claim_end_date   else None,
            }
            for i, code in enumerate(diag_codes, start=2):
                payload[f"diagnosis_code_{i}"] = code.strip() or None
            for i, code in enumerate(proc_codes, start=1):
                payload[f"procedure_code_{i}"] = code.strip() or None
            payload["attending_physician"] = attending_physician.strip() or None
            payload["operating_physician"] = operating_physician.strip() or None
            payload["other_physician"]     = other_physician.strip() or None
            payload.update(chronic_values)

            try:
                with st.spinner("Submitting claim..."):
                    resp = requests.post(INGESTION_URL, json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"✅ Claim **{data.get('claim_id')}** submitted successfully!\n\n"
                        f"Status: {data.get('status')} | Queued at: {data.get('queued_at')}"
                    )
                    st.info("Claim queued for fraud analysis. Results appear in Claims List shortly.")
                else:
                    detail = resp.json().get("detail", resp.text)
                    st.error(f"❌ Submission failed ({resp.status_code}): {detail}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach the ingestion API. Make sure the service is running.")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out.")
            except Exception as ex:
                st.error(f"❌ Unexpected error: {ex}")


# ── Analytics page ─────────────────────────────────────────────────────────────
def show_analytics():
    st.header("📈 Analytics")

    try:
        with get_session() as session:
            total_claims    = session.query(func.count(Claim.id)).scalar() or 0
            total_analyzed  = session.query(func.count(FraudAnalysis.id)).scalar() or 0
            total_anomalies = (session.query(func.count(FraudAnalysis.id))
                               .filter(FraudAnalysis.is_anomaly == True).scalar() or 0)
            avg_amount      = session.query(func.avg(Claim.claim_amount)).scalar() or 0
            avg_score       = session.query(func.avg(FraudAnalysis.anomaly_score)).scalar() or 0

            fraud_rate = (total_anomalies / total_analyzed * 100) if total_analyzed > 0 else 0

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Total Claims",      f"{total_claims:,}")
            k2.metric("Analyzed",          f"{total_analyzed:,}")
            k3.metric("Anomalies",         f"{total_anomalies:,}")
            k4.metric("Fraud Rate",        f"{fraud_rate:.1f}%")
            k5.metric("Avg Claim Amount",  f"${float(avg_amount):,.0f}")

            st.markdown("---")

            if total_analyzed == 0:
                st.info("No analyzed claims yet.")
                return

            # ── Row 1: Time-series + Risk breakdown ────────────────────────────
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Claims per Day (last 30 days)")
                from sqlalchemy import text, Integer as SAInteger
                from datetime import timedelta
                cutoff = datetime.utcnow() - timedelta(days=30)

                ts_rows = (
                    session.query(
                        cast(Claim.created_at, Date).label("day"),
                        func.count(Claim.id).label("total"),
                        func.count(FraudAnalysis.claim_id)
                            .filter(FraudAnalysis.is_anomaly == True).label("anomalies"),
                    )
                    .outerjoin(FraudAnalysis, Claim.claim_id == FraudAnalysis.claim_id)
                    .filter(Claim.created_at >= cutoff)
                    .group_by("day")
                    .order_by("day")
                    .all()
                )
                if ts_rows:
                    ts_df = pd.DataFrame(ts_rows, columns=["day", "total", "anomalies"])
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Bar(
                        x=ts_df["day"], y=ts_df["total"],
                        name="Total Claims", marker_color="#4C9BE8"))
                    fig_ts.add_trace(go.Scatter(
                        x=ts_df["day"], y=ts_df["anomalies"],
                        name="Anomalies", mode="lines+markers",
                        line=dict(color="#FF6B6B", width=2)))
                    fig_ts.update_layout(legend=dict(orientation="h"), margin=dict(t=10))
                    st.plotly_chart(fig_ts, use_container_width=True)
                else:
                    st.info("No claims in the last 30 days")

            with col2:
                st.subheader("Risk Level Breakdown")
                risk_rows = (
                    session.query(FraudAnalysis.risk_level,
                                  func.count(FraudAnalysis.id).label("cnt"))
                    .group_by(FraudAnalysis.risk_level)
                    .order_by(func.count(FraudAnalysis.id).desc())
                    .all()
                )
                if risk_rows:
                    risk_df = pd.DataFrame(risk_rows, columns=["risk_level", "cnt"])
                    color_map = {
                        "CRITICAL": "#D32F2F", "HIGH": "#FF6B6B",
                        "MEDIUM":   "#FFA726", "LOW":  "#66BB6A",
                    }
                    fig_bar = px.bar(
                        risk_df, x="risk_level", y="cnt",
                        color="risk_level", color_discrete_map=color_map,
                        labels={"risk_level": "Risk Level", "cnt": "Count"},
                    )
                    fig_bar.update_layout(showlegend=False, margin=dict(t=10))
                    st.plotly_chart(fig_bar, use_container_width=True)

            # ── Row 2: Anomaly score distribution + Claim amount by risk ───────
            col3, col4 = st.columns(2)

            with col3:
                st.subheader("Anomaly Score Distribution")
                score_rows = (
                    session.query(FraudAnalysis.anomaly_score,
                                  FraudAnalysis.risk_level).all()
                )
                if score_rows:
                    score_df = pd.DataFrame(score_rows, columns=["anomaly_score", "risk_level"])
                    fig_hist = px.histogram(
                        score_df, x="anomaly_score", color="risk_level",
                        nbins=40, barmode="overlay",
                        color_discrete_map={
                            "CRITICAL": "#D32F2F", "HIGH": "#FF6B6B",
                            "MEDIUM": "#FFA726",   "LOW":  "#66BB6A",
                        },
                        labels={"anomaly_score": "Anomaly Score"},
                    )
                    fig_hist.update_layout(margin=dict(t=10))
                    st.plotly_chart(fig_hist, use_container_width=True)

            with col4:
                st.subheader("Avg Claim Amount by Risk Level")
                amt_rows = (
                    session.query(
                        FraudAnalysis.risk_level,
                        func.avg(Claim.claim_amount).label("avg_amount"),
                    )
                    .join(Claim, Claim.claim_id == FraudAnalysis.claim_id)
                    .group_by(FraudAnalysis.risk_level)
                    .all()
                )
                if amt_rows:
                    amt_df = pd.DataFrame(amt_rows, columns=["risk_level", "avg_amount"])
                    amt_df["avg_amount"] = amt_df["avg_amount"].astype(float)
                    fig_amt = px.bar(
                        amt_df, x="risk_level", y="avg_amount",
                        color="risk_level",
                        color_discrete_map={
                            "CRITICAL": "#D32F2F", "HIGH": "#FF6B6B",
                            "MEDIUM": "#FFA726",   "LOW":  "#66BB6A",
                        },
                        labels={"risk_level": "Risk Level", "avg_amount": "Avg Amount ($)"},
                    )
                    fig_amt.update_layout(showlegend=False, margin=dict(t=10))
                    st.plotly_chart(fig_amt, use_container_width=True)

    except Exception as e:
        st.error(f"Analytics error: {e}")


# ── Entry point ─────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.get("authenticated"):
        login_page()
    else:
        main_dashboard()


if __name__ == "__main__":
    main()
