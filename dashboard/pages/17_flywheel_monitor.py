"""
Data Flywheel Monitor
=====================

Monitor and manage the continuous learning flywheel.

Features:
- Model registry with version control
- A/B test monitoring
- Feedback collection stats
- Distillation queue and status
- Agent evaluation metrics
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Flywheel Monitor | Cognisom",
    page_icon="🔄",
    layout="wide",
)

# Auth gate
try:
    from cognisom.auth.middleware import streamlit_page_gate
    user = streamlit_page_gate(required_tier="institution")
except Exception:
    user = None

st.title("🔄 Data Flywheel Monitor")
st.markdown("Continuous learning, model optimization, and feedback collection")

# ═══════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🗂️ Model Registry", "⚗️ A/B Tests", "💬 Feedback", "📈 Agent Metrics"
])

# ─────────────────────────────────────────────────────────────────────────
# Tab 1: Overview
# ─────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Flywheel Status")

    # Try to load flywheel
    try:
        from cognisom.flywheel.flywheel import DataFlywheel, FlywheelConfig, FlywheelStatus
        from cognisom.flywheel.model_registry import ModelRegistry
        from cognisom.flywheel.feedback import FeedbackCollector
        from cognisom.eval.agent_eval import AgentEvaluator

        flywheel_available = True
    except ImportError:
        flywheel_available = False

    if flywheel_available:
        # Initialize components
        if "flywheel" not in st.session_state:
            config = FlywheelConfig(data_dir="data/flywheel")
            st.session_state.flywheel = DataFlywheel(config)

        flywheel = st.session_state.flywheel

        # Status cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            status = flywheel.status.value if hasattr(flywheel, 'status') else "idle"
            status_colors = {"running": "🟢", "idle": "🟡", "error": "🔴", "training": "🔵"}
            st.metric("Flywheel Status", f"{status_colors.get(status, '⚪')} {status.title()}")

        with col2:
            try:
                registry = ModelRegistry(registry_dir="data/model_registry")
                versions = registry.list_versions()
                st.metric("Model Versions", len(versions))
            except:
                st.metric("Model Versions", 0)

        with col3:
            try:
                collector = FeedbackCollector(storage_dir="data/feedback")
                summary = collector.get_summary()
                st.metric("Total Feedback", summary.total_feedback)
            except:
                st.metric("Total Feedback", 0)

        with col4:
            try:
                evaluator = AgentEvaluator(storage_dir="data/agent_interactions")
                # Would get from evaluator
                st.metric("Interactions Today", "—")
            except:
                st.metric("Interactions Today", 0)

        st.divider()

        # Flywheel diagram
        st.markdown("### Flywheel Pipeline")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            ```
            ┌─────────────────────────────────────────────────────────────┐
            │                     DATA FLYWHEEL                          │
            │                                                             │
            │    ┌──────────┐     ┌──────────┐     ┌──────────┐         │
            │    │  Agent   │────▶│ Evaluate │────▶│ Collect  │         │
            │    │Interact. │     │ Quality  │     │ Feedback │         │
            │    └──────────┘     └──────────┘     └──────────┘         │
            │         ▲                                  │               │
            │         │                                  ▼               │
            │    ┌──────────┐     ┌──────────┐     ┌──────────┐         │
            │    │  Deploy  │◀────│ A/B Test │◀────│ Distill  │         │
            │    │  Model   │     │          │     │  Model   │         │
            │    └──────────┘     └──────────┘     └──────────┘         │
            │                                                             │
            └─────────────────────────────────────────────────────────────┘
            ```
            """)

        with col2:
            st.markdown("**Pipeline Steps:**")
            st.markdown("""
            1. **Capture** — Log agent interactions
            2. **Evaluate** — Score quality (auto + manual)
            3. **Collect** — Gather human feedback
            4. **Distill** — Train smaller models
            5. **A/B Test** — Compare model versions
            6. **Deploy** — Promote winning models
            """)

        st.divider()

        # Recent activity
        st.markdown("### Recent Activity")

        # Mock activity data
        activities = [
            {"time": "2 min ago", "event": "New feedback received", "type": "rating", "details": "5/5 stars"},
            {"time": "15 min ago", "event": "Model v2.1 deployed", "type": "deploy", "details": "10% traffic"},
            {"time": "1 hour ago", "event": "Distillation completed", "type": "training", "details": "Loss: 0.023"},
            {"time": "3 hours ago", "event": "A/B test started", "type": "test", "details": "v2.0 vs v2.1"},
            {"time": "6 hours ago", "event": "Quality report generated", "type": "eval", "details": "87% quality"},
        ]

        for activity in activities:
            type_icons = {"rating": "⭐", "deploy": "🚀", "training": "🎓", "test": "⚗️", "eval": "📊"}
            icon = type_icons.get(activity["type"], "📌")
            st.text(f"{icon} [{activity['time']}] {activity['event']} — {activity['details']}")

    else:
        st.warning("Flywheel modules not available")

# ─────────────────────────────────────────────────────────────────────────
# Tab 2: Model Registry
# ─────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Model Registry")
    st.markdown("Version control and deployment management for distilled models.")

    try:
        from cognisom.flywheel.model_registry import ModelRegistry, ModelStatus

        registry = ModelRegistry(registry_dir="data/model_registry")

        col1, col2 = st.columns([2, 1])

        with col1:
            versions = registry.list_versions()

            if versions:
                # Build table data
                table_data = []
                for v in versions:
                    table_data.append({
                        "Version": v.version_id,
                        "Base Model": v.base_model,
                        "Status": v.status.value if hasattr(v.status, 'value') else str(v.status),
                        "Traffic %": v.traffic_percent,
                        "Accuracy": v.metrics.get("accuracy", "—") if v.metrics else "—",
                        "Created": v.created_at[:16] if isinstance(v.created_at, str) else "—",
                    })

                st.dataframe(table_data, use_container_width=True)

                # Actions
                st.markdown("### Actions")

                selected_version = st.selectbox(
                    "Select Version",
                    [v.version_id for v in versions]
                )

                col_a1, col_a2, col_a3 = st.columns(3)

                with col_a1:
                    traffic = st.slider("Traffic %", 0, 100, 10)

                with col_a2:
                    ab_test = st.checkbox("Start A/B Test")

                with col_a3:
                    if st.button("🚀 Deploy", type="primary"):
                        try:
                            registry.deploy(selected_version, traffic_percent=traffic, ab_test=ab_test)
                            st.success(f"Deployed {selected_version} with {traffic}% traffic")
                        except Exception as e:
                            st.error(f"Deploy failed: {e}")

            else:
                st.info("No models registered yet")

                # Register new model form
                st.markdown("### Register New Model")

                with st.form("register_model"):
                    model_path = st.text_input("Model Path", "/models/nemotron-8b-distilled")
                    base_model = st.text_input("Base Model", "nemotron-8b")
                    accuracy = st.number_input("Accuracy", 0.0, 1.0, 0.85)
                    latency = st.number_input("Latency (ms)", 0, 1000, 150)
                    tags = st.text_input("Tags (comma-separated)", "distilled,lora")

                    if st.form_submit_button("Register Model"):
                        try:
                            version = registry.register(
                                model_path=model_path,
                                base_model=base_model,
                                metrics={"accuracy": accuracy, "latency_ms": latency},
                                tags=tags.split(",") if tags else [],
                            )
                            st.success(f"Registered: {version.version_id}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Registration failed: {e}")

        with col2:
            st.markdown("### Registry Stats")

            if versions:
                # Count by status
                status_counts = {}
                for v in versions:
                    status = v.status.value if hasattr(v.status, 'value') else str(v.status)
                    status_counts[status] = status_counts.get(status, 0) + 1

                fig = go.Figure(data=[go.Pie(
                    labels=list(status_counts.keys()),
                    values=list(status_counts.values()),
                    hole=0.4,
                )])
                fig.update_layout(title="Models by Status", height=250, margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

                # Traffic distribution
                traffic_versions = [(v.version_id, v.traffic_percent) for v in versions if v.traffic_percent > 0]
                if traffic_versions:
                    fig2 = go.Figure(data=[go.Bar(
                        x=[t[0] for t in traffic_versions],
                        y=[t[1] for t in traffic_versions],
                        marker_color="green",
                    )])
                    fig2.update_layout(title="Traffic Distribution", height=200, margin=dict(t=40, b=20))
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No data to display")

    except ImportError as e:
        st.error(f"Model registry not available: {e}")

# ─────────────────────────────────────────────────────────────────────────
# Tab 3: A/B Tests
# ─────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("A/B Testing")
    st.markdown("Compare model performance in production.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Active Tests")

        # Mock A/B test data
        active_tests = [
            {
                "name": "v2.0 vs v2.1",
                "started": "2026-02-01",
                "control": "v2.0 (90%)",
                "treatment": "v2.1 (10%)",
                "metric": "Quality Score",
                "control_value": 0.82,
                "treatment_value": 0.88,
                "significance": 0.04,
            }
        ]

        for test in active_tests:
            with st.expander(f"🧪 {test['name']}", expanded=True):
                st.markdown(f"**Started:** {test['started']}")
                st.markdown(f"**Control:** {test['control']}")
                st.markdown(f"**Treatment:** {test['treatment']}")

                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    st.metric("Control", f"{test['control_value']:.2f}")
                with col_t2:
                    st.metric("Treatment", f"{test['treatment_value']:.2f}",
                             delta=f"+{(test['treatment_value'] - test['control_value']):.2f}")
                with col_t3:
                    sig = "✅ Significant" if test['significance'] < 0.05 else "⏳ Not yet"
                    st.metric("p-value", f"{test['significance']:.3f}")
                    st.caption(sig)

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("🏆 Promote Treatment", key=f"promote_{test['name']}"):
                        st.success("Treatment promoted to 100% traffic")
                with col_btn2:
                    if st.button("🛑 Stop Test", key=f"stop_{test['name']}"):
                        st.info("Test stopped, reverted to control")

    with col2:
        st.markdown("### Test Results Over Time")

        # Mock time series
        days = 14
        dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]
        control_scores = [0.80 + np.random.normal(0, 0.02) for _ in range(days)]
        treatment_scores = [0.85 + np.random.normal(0, 0.02) for _ in range(days)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=control_scores,
            name="Control (v2.0)",
            line=dict(color="blue", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=treatment_scores,
            name="Treatment (v2.1)",
            line=dict(color="green", width=2),
        ))
        fig.update_layout(
            title="Quality Score Over Time",
            xaxis_title="Date",
            yaxis_title="Quality Score",
            height=350,
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Sample size
        st.markdown("### Sample Size")
        st.metric("Control Samples", "4,521")
        st.metric("Treatment Samples", "502")

        # Minimum detectable effect
        st.caption("MDE: 5% with 80% power at current sample size")

# ─────────────────────────────────────────────────────────────────────────
# Tab 4: Feedback
# ─────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Feedback Collection")
    st.markdown("Human feedback for RLHF training data.")

    try:
        from cognisom.flywheel.feedback import FeedbackCollector

        collector = FeedbackCollector(storage_dir="data/feedback")
        summary = collector.get_summary()

        col1, col2 = st.columns([1, 1])

        with col1:
            # Summary metrics
            st.markdown("### Summary")

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Total Feedback", summary.total_feedback)
            with col_m2:
                st.metric("Avg Rating", f"{summary.average_rating:.1f}/5")
            with col_m3:
                st.metric("Thumbs Up %", f"{summary.thumbs_up_pct:.0f}%")

            col_m4, col_m5 = st.columns(2)
            with col_m4:
                st.metric("Corrections", summary.corrections_count)
            with col_m5:
                st.metric("Preferences", summary.preferences_count)

            st.divider()

            # Submit test feedback
            st.markdown("### Submit Feedback")

            with st.form("submit_feedback"):
                interaction_id = st.text_input("Interaction ID", "int_test_001")
                feedback_type = st.selectbox("Type", ["Rating", "Thumbs", "Correction", "Preference"])

                if feedback_type == "Rating":
                    rating = st.slider("Rating", 1, 5, 4)
                    comment = st.text_area("Comment (optional)")
                elif feedback_type == "Thumbs":
                    thumbs_up = st.checkbox("Thumbs Up", value=True)
                    comment = st.text_area("Comment (optional)")
                elif feedback_type == "Correction":
                    original = st.text_area("Original Response")
                    corrected = st.text_area("Corrected Response")
                    comment = st.text_area("Comment")
                else:  # Preference
                    response_a = st.text_area("Response A")
                    response_b = st.text_area("Response B")
                    preferred = st.radio("Preferred", ["a", "b"], horizontal=True)

                if st.form_submit_button("Submit"):
                    try:
                        if feedback_type == "Rating":
                            collector.submit_rating(interaction_id, rating, comment or None)
                        elif feedback_type == "Thumbs":
                            collector.submit_thumbs(interaction_id, thumbs_up, comment or None)
                        elif feedback_type == "Correction":
                            collector.submit_correction(interaction_id, original, corrected, comment)
                        else:
                            collector.submit_preference(interaction_id, response_a, response_b, preferred)
                        st.success("Feedback submitted!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        with col2:
            st.markdown("### Rating Distribution")

            # Mock rating distribution
            ratings = [1, 2, 3, 4, 5]
            counts = [5, 12, 45, 120, 200]

            fig = go.Figure(data=[go.Bar(
                x=ratings,
                y=counts,
                marker_color=["red", "orange", "yellow", "lightgreen", "green"],
            )])
            fig.update_layout(
                xaxis_title="Rating",
                yaxis_title="Count",
                height=250,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Training Data Ready")

            corrections = collector.get_corrections_for_training()
            preferences = collector.get_preferences_for_rlhf()

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.metric("SFT Examples", len(corrections))
            with col_d2:
                st.metric("RLHF Pairs", len(preferences))

            if st.button("📥 Export Training Data"):
                import json
                data = {
                    "corrections": corrections,
                    "preferences": preferences,
                }
                st.download_button(
                    "Download JSON",
                    data=json.dumps(data, indent=2),
                    file_name="rlhf_training_data.json",
                    mime="application/json",
                )

    except ImportError as e:
        st.error(f"Feedback module not available: {e}")

# ─────────────────────────────────────────────────────────────────────────
# Tab 5: Agent Metrics
# ─────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Agent Evaluation Metrics")
    st.markdown("Quality metrics for AI agent interactions.")

    try:
        from cognisom.eval.agent_eval import AgentEvaluator

        evaluator = AgentEvaluator(storage_dir="data/agent_interactions")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Capture New Interaction")

            with st.form("capture_interaction"):
                query = st.text_area("Query", "What is the role of AR in prostate cancer?")
                response = st.text_area("Response", "The androgen receptor (AR) plays a central role...")
                agent_type = st.selectbox("Agent Type", ["researcher", "assistant", "discovery"])
                latency = st.number_input("Latency (ms)", 0, 5000, 250)

                if st.form_submit_button("Capture & Evaluate"):
                    try:
                        interaction = evaluator.capture(
                            query=query,
                            response=response,
                            agent_type=agent_type,
                            latency_ms=latency,
                        )
                        st.success("Interaction captured!")

                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1:
                            st.metric("Relevance", f"{interaction.relevance_score:.2f}")
                        with col_s2:
                            st.metric("Helpfulness", f"{interaction.helpfulness_score:.2f}")
                        with col_s3:
                            st.metric("Factuality", f"{interaction.factuality_score:.2f}")

                    except Exception as e:
                        st.error(f"Error: {e}")

            st.divider()

            st.markdown("### Batch Evaluation")

            if st.button("📊 Run Batch Evaluation"):
                try:
                    report = evaluator.evaluate_batch()
                    st.success("Batch evaluation complete!")

                    st.metric("Interactions Evaluated", report.interactions_evaluated)
                    st.metric("Overall Quality", f"{report.overall_quality:.1f}/100")
                    st.metric("Mean Relevance", f"{report.mean_relevance:.2f}")
                    st.metric("Mean Helpfulness", f"{report.mean_helpfulness:.2f}")
                    st.metric("Distillation Candidates", report.distillation_candidates)

                except Exception as e:
                    st.error(f"Error: {e}")

        with col2:
            st.markdown("### Quality Over Time")

            # Mock time series
            days = 30
            dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]
            quality = [75 + np.cumsum(np.random.normal(0.2, 1, 1))[0] for _ in range(days)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=quality,
                fill='tozeroy',
                line=dict(color="purple", width=2),
            ))
            fig.update_layout(
                title="Overall Quality Score",
                xaxis_title="Date",
                yaxis_title="Quality (0-100)",
                height=250,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Quality by Agent Type")

            agent_types = ["researcher", "assistant", "discovery"]
            scores = [85, 78, 82]

            fig2 = go.Figure(data=[go.Bar(
                x=agent_types,
                y=scores,
                marker_color=["blue", "green", "orange"],
            )])
            fig2.update_layout(
                yaxis_title="Quality Score",
                height=200,
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Get distillation data
            st.markdown("### Distillation Data")
            distill_data = evaluator.get_distillation_data(min_quality=0.7)
            st.metric("High-Quality Examples", len(distill_data))

            if distill_data:
                if st.button("📥 Export for Training"):
                    import json
                    st.download_button(
                        "Download JSON",
                        data=json.dumps(distill_data, indent=2, default=str),
                        file_name="distillation_data.json",
                        mime="application/json",
                    )

    except ImportError as e:
        st.error(f"Agent eval module not available: {e}")

# Footer
from cognisom.dashboard.footer import render_footer
render_footer()
