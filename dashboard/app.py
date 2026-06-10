import streamlit as st
import os
import glob
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

st.set_page_config(
    page_title="YOLO Generalization Benchmark",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR     = '/home/sdvn_hinder_metric/Sahan/ROS2-YOLO-Robot-Perception-Generalization'
MATRIX_CSV   = f'{BASE_DIR}/benchmark/results/matrix_results.csv'
FINETUNE_CSV = f'{BASE_DIR}/benchmark/results/finetune_results.csv'
PLOTS_DIR    = f'{BASE_DIR}/benchmark/results/plots'
DATASETS_DIR = f'{BASE_DIR}/datasets'

sys.path.append(BASE_DIR)
try:
    from benchmark.scoring import compute_scores
except ImportError:
    compute_scores = None

def sidebar():
    st.sidebar.title("🤖 YOLO Generalization")
    st.sidebar.subheader("Synthetic Environment Impact Study")
    st.sidebar.divider()
    
    page = st.sidebar.radio("Navigation", 
        ["📊 Overview", "🔬 Benchmark Matrix", "🔁 Fine-tuning", "🎯 Deployment Scorer"]
    )
    
    st.sidebar.divider()
    st.sidebar.markdown("### Dataset Status")
    
    envs = ['env_a', 'env_b', 'env_c']
    for e in envs:
        env_dir = os.path.join(DATASETS_DIR, e)
        if os.path.isdir(env_dir):
            img_count = len(glob.glob(os.path.join(env_dir, '**', '*.jpg'), recursive=True))
        else:
            img_count = 0
            
        status = "✅" if img_count > 0 else "⏳"
        display_name = e.replace('_', ' ').title()
        st.sidebar.write(f"{display_name}: {img_count} images {status}")
        
    return page

def page_overview():
    st.title("📊 Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    envs = [e for e in ['env_a', 'env_b', 'env_c'] if os.path.isdir(os.path.join(DATASETS_DIR, e))]
    col1.metric("Environments", str(len(envs)), "Sparse / Warehouse / Corridor")
    
    models_dir = os.path.join(BASE_DIR, 'training', 'runs')
    models_count = len(glob.glob(os.path.join(models_dir, '*', 'weights', 'best.pt')))
    col2.metric("Models Trained", str(models_count))
    
    col3.metric("Benchmark Combos", "9", "3 models × 3 environments")
    
    total_imgs = len(glob.glob(os.path.join(DATASETS_DIR, '**', '*.jpg'), recursive=True))
    col4.metric("Dataset Images", str(total_imgs))
    
    st.info("This study benchmarks how training environment complexity affects YOLOv8 generalization. Three models (trained on Env A, B, and Mixed) are cross-evaluated across all environments to measure domain shift.")
    
    st.markdown("> **Research Question:** How does synthetic training environment structure affect object detection generalization under domain shift?")
    
    st.subheader("Project Timeline")
    st.markdown("""
| Week | Status | Task |
|---|---|---|
| Week 1 | ✅ | ROS2 + Gazebo + Repo Init |
| Week 2 | ✅ | Auto-labeling Data Pipeline |
| Week 3 | ✅ | Dataset Generation |
| Week 4 | ⏳ | YOLO Training & Finetuning |
| Week 5 | ⏳ | Cross-Evaluation Benchmark |
| Week 6 | ⏳ | Scoring Dashboard |
| Week 7 | ⏳ | Real-world Deployment |
| Week 8 | ⏳ | IEEE Paper Submission |
    """)

def page_benchmark():
    st.title("🔬 Benchmark Matrix")
    
    if not os.path.exists(MATRIX_CSV):
        st.warning("⏳ Benchmark not yet run. Train models first, then run: python3 benchmark/run_matrix.py")
        st.stop()
        
    df = pd.read_csv(MATRIX_CSV)
    
    st.subheader("Interactive mAP50 Heatmap")
    pivot_df = df.pivot(index='model', columns='test_env', values='mAP50')
    
    models_order = ['model_a', 'model_b', 'model_mixed']
    envs_order = ['env_a', 'env_b', 'env_c']
    
    pivot_df = pivot_df.reindex(index=[m for m in models_order if m in pivot_df.index], 
                                columns=[e for e in envs_order if e in pivot_df.columns])
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=['Test Env A', 'Test Env B', 'Test Env C'],
        y=['Model A', 'Model B', 'Model Mixed'],
        colorscale='RdYlGn',
        text=pivot_df.values,
        texttemplate='%{text:.3f}',
        colorbar=dict(title='mAP@50'),
        zmin=0, zmax=1
    ))
    fig.update_layout(
        title='mAP@50 Cross-Evaluation Matrix',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Domain Shift Analysis")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_bar = go.Figure()
        for model in pivot_df.index:
            fig_bar.add_trace(go.Bar(
                name=model,
                x=pivot_df.columns,
                y=pivot_df.loc[model].values
            ))
        fig_bar.update_layout(barmode='group', title="mAP@50 per Model per Environment")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col2:
        avg_maps = pivot_df.mean(axis=1)
        best_avg_model = avg_maps.idxmax()
        best_avg_val = avg_maps.max()
        
        max_drop = 0
        drop_model = ""
        for m in pivot_df.index:
            baseline_env = 'env_a' if m == 'model_a' else ('env_b' if m == 'model_b' else None)
            if baseline_env and baseline_env in pivot_df.columns:
                base_val = pivot_df.loc[m, baseline_env]
                for e in pivot_df.columns:
                    drop = base_val - pivot_df.loc[m, e]
                    if drop > max_drop:
                        max_drop = drop
                        drop_model = m
                        
        best_cell_val = pivot_df.max().max()
        best_cell_model = ""
        best_cell_env = ""
        for m in pivot_df.index:
            for e in pivot_df.columns:
                if pivot_df.loc[m, e] == best_cell_val:
                    best_cell_model = m
                    best_cell_env = e
                    
        st.info(f"**Key Findings:**\n\n"
                f"- **{best_avg_model.replace('_',' ').title()}** achieved highest average mAP of {best_avg_val:.3f}\n\n"
                f"- **Largest domain shift:** {drop_model.replace('_',' ').title()} dropped {max_drop*100:.1f}% from baseline\n\n"
                f"- **Best single-env model:** {best_cell_model.replace('_',' ').title()} on {best_cell_env.replace('_',' ').title()} (mAP: {best_cell_val:.3f})")
                
    st.subheader("Raw Results Table")
    
    def highlight_baseline(row):
        return ['background-color: rgba(144, 238, 144, 0.2)' if row['is_baseline'] else '' for _ in row]
        
    formatted_df = df.copy()
    for col in ['mAP50', 'precision', 'recall']:
        formatted_df[col] = formatted_df[col].map('{:.3f}'.format)
        
    st.dataframe(formatted_df.style.apply(highlight_baseline, axis=1), use_container_width=True)
    
    st.subheader("Static Plots")
    col1, col2 = st.columns(2)
    p1 = os.path.join(PLOTS_DIR, 'heatmap_mAP50.png')
    p2 = os.path.join(PLOTS_DIR, 'heatmap_domain_shift.png')
    
    with col1:
        if os.path.exists(p1):
            st.image(p1, use_column_width=True)
    with col2:
        if os.path.exists(p2):
            st.image(p2, use_column_width=True)

def page_finetuning():
    st.title("🔁 Fine-tuning Experiment")
    
    if not os.path.exists(FINETUNE_CSV):
        st.warning("⏳ Run: python3 training/finetune_experiment.py")
        st.stop()
        
    st.info("Few-shot domain adaptation: Can model_b (trained on Env B) adapt to Env A using only 50 images? Compared against training from scratch.")
    
    df = pd.read_csv(FINETUNE_CSV)
    
    base_val = df.loc[df['condition'] == 'baseline_no_adapt', 'mAP50'].values[0] if not df.loc[df['condition'] == 'baseline_no_adapt'].empty else 0
    fine_val = df.loc[df['condition'] == 'finetuned_50shots', 'mAP50'].values[0] if not df.loc[df['condition'] == 'finetuned_50shots'].empty else 0
    scratch_val = df.loc[df['condition'] == 'scratch_50shots', 'mAP50'].values[0] if not df.loc[df['condition'] == 'scratch_50shots'].empty else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline (no adapt)", f"{base_val:.3f}", f"{(base_val - scratch_val):.3f} vs scratch")
    col2.metric("Fine-tuned 50 shots", f"{fine_val:.3f}", f"{(fine_val - base_val):.3f} vs baseline")
    col3.metric("Scratch 50 shots", f"{scratch_val:.3f}")
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("mAP@50", "FPS"))
    
    conds = ['baseline_no_adapt', 'finetuned_50shots', 'scratch_50shots']
    colors = ['red', 'green', 'blue']
    
    maps = [df.loc[df['condition'] == c, 'mAP50'].values[0] if not df.loc[df['condition'] == c].empty else 0 for c in conds]
    fpss = [df.loc[df['condition'] == c, 'fps'].values[0] if not df.loc[df['condition'] == c].empty else 0 for c in conds]
    
    fig.add_trace(go.Bar(x=conds, y=maps, marker_color=colors, showlegend=False), row=1, col=1)
    if base_val > 0:
        fig.add_hline(y=base_val, line_dash="dash", line_color="red", row=1, col=1)
    
    fig.add_trace(go.Bar(x=conds, y=fpss, marker_color=colors, showlegend=False), row=1, col=2)
    
    fig.update_layout(title_text="Few-Shot Domain Adaptation Results")
    st.plotly_chart(fig, use_container_width=True)
    
    improvement = fine_val - base_val
    better_scratch = (fine_val / scratch_val - 1) if scratch_val > 0 else 0
    st.success(f"✅ Fine-tuning recovered {improvement:.1%} mAP over baseline using only 50 images — {better_scratch:.1%} better than training from scratch.")

def page_deployment():
    st.title("🎯 Deployment Scorer")
    
    results_file = MATRIX_CSV
    if not os.path.exists(MATRIX_CSV):
        st.warning("Using example data - run benchmark matrix to use real data")
        os.makedirs(os.path.dirname(MATRIX_CSV), exist_ok=True)
        import csv
        with open(MATRIX_CSV, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'test_env', 'mAP50', 'precision', 'recall', 'fps', 'is_baseline'])
            writer.writerow(['model_a', 'env_a', '0.8', '0.8', '0.8', '30', 'True'])
            writer.writerow(['model_a', 'env_b', '0.4', '0.4', '0.4', '30', 'False'])
            writer.writerow(['model_b', 'env_a', '0.3', '0.3', '0.3', '30', 'False'])
            writer.writerow(['model_b', 'env_b', '0.9', '0.9', '0.9', '30', 'True'])
            writer.writerow(['model_mixed', 'env_a', '0.7', '0.7', '0.7', '28', 'False'])
            writer.writerow(['model_mixed', 'env_b', '0.75', '0.75', '0.75', '28', 'False'])
            
    st.subheader("Configure Deployment Priority")
    col1, col2, col3 = st.columns(3)
    w1 = col1.slider("Accuracy weight (mAP)", 0.0, 1.0, 0.5, 0.1)
    w2 = col2.slider("Robustness weight", 0.0, 1.0, 0.3, 0.1)
    w3 = col3.slider("Speed weight (FPS)", 0.0, 1.0, 0.2, 0.1)
    
    if abs((w1 + w2 + w3) - 1.0) > 0.01:
        st.warning(f"Weights sum to {w1+w2+w3:.1f} — recommend they sum to 1.0")
        
    target_env = st.selectbox("Target deployment environment", ["Any", "env_a (Sparse)", "env_b (Warehouse)", "env_c (Corridor)"])
    if target_env == "Any":
        env_arg = None
    else:
        env_arg = target_env.split(" ")[0]
        
    if compute_scores:
        scores = compute_scores(results_file, target_env=env_arg, w1=w1, w2=w2, w3=w3)
        
        medals = ["🥇", "🥈", "🥉"]
        for i, s in enumerate(scores):
            with st.container(border=True):
                medal = medals[i] if i < 3 else "🏅"
                st.markdown(f"### {medal} {s['model'].replace('_', ' ').title():<12} Score: {s['score']:.3f}")
                st.write(f"mAP50: {s['mAP50']:.3f} | Robustness: {s['robustness']:.3f} | FPS: {s['avg_fps']:.1f}")
    else:
        st.error("Failed to import compute_scores from benchmark.scoring")
        
    st.latex(r"score = w_1 \times mAP_{50} + w_2 \times (1 - \sigma_{shift}) + w_3 \times FPS_{norm}")

def main():
    page = sidebar()
    
    if page == "📊 Overview":
        page_overview()
    elif page == "🔬 Benchmark Matrix":
        page_benchmark()
    elif page == "🔁 Fine-tuning":
        page_finetuning()
    elif page == "🎯 Deployment Scorer":
        page_deployment()

if __name__ == "__main__":
    main()
