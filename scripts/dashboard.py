#!/usr/bin/env python3
"""Interactive Streamlit dashboard for experiment tracking and analysis.

This dashboard provides comprehensive visualization and analysis of multi-agent
experiment configurations and performance results from timestamped result directories.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
from pathlib import Path
import sys
import os
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure page
st.set_page_config(
    page_title="ConvFinQA Experiment Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

def parse_timestamp_from_directory(dir_name: str) -> datetime:
    """Parse timestamp from directory name format YYYYMMDD_HHMMSS."""
    try:
        return datetime.strptime(dir_name, "%Y%m%d_%H%M%S")
    except ValueError:
        return datetime.min

def load_experiment_data():
    """Load experiment data from timestamped result directories."""
    results_dir = Path("experiment_tracking/results")
    
    if not results_dir.exists():
        return pd.DataFrame()
    
    experiments = []
    
    # Get all experiment directories
    experiment_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    experiment_dirs.sort(key=lambda x: x.name)
    
    for exp_dir in experiment_dirs:
        metadata_file = exp_dir / "run_metadata.txt"
        all_results_file = exp_dir / "all_results.txt"
        
        if not (metadata_file.exists() and all_results_file.exists()):
            continue
            
        try:
            # Parse metadata
            metadata = parse_metadata_file(metadata_file)
            
            # Parse detailed results
            detailed_results = parse_all_results_file(all_results_file)
            
            # Parse timestamp from directory name
            timestamp = parse_timestamp_from_directory(exp_dir.name)
            
            experiment = {
                'experiment_id': exp_dir.name,
                'timestamp': timestamp,
                'directory': str(exp_dir),
                **metadata,
                'detailed_results': detailed_results
            }
            
            experiments.append(experiment)
            
        except Exception as e:
            st.warning(f"Failed to parse experiment {exp_dir.name}: {e}")
            continue
    
    return pd.DataFrame(experiments)

def parse_metadata_file(metadata_file: Path) -> dict:
    """Parse run_metadata.txt file to extract experiment configuration."""
    metadata = {}
    
    with open(metadata_file, 'r') as f:
        content = f.read()
    
    # Extract key metrics using regex
    patterns = {
        'accuracy': r'CrewAI Multi-Agent: ([\d.]+)%',
        'total_cost': r'Total cost: \$([\d.]+)',
        'cost_per_question': r'Cost per question: \$([\d.]+)',
        'total_questions': r'Total questions: (\d+)',
        'passed_questions': r'Passed questions: (\d+)',
        'failed_questions': r'Failed questions: (\d+)',
        'conversations_evaluated': r'Conversations evaluated: (\d+)',
        'sampling_strategy': r'Sampling strategy: (\w+)',
        'random_seed': r'Random seed: (\d+)',
        'config_hash': r'Configuration Hash: (\w+)',
        'agent_models': r'Agent Models: (.+)',
        'target_achievement': r'Target Achievement: (.+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            value = match.group(1)
            # Convert numeric values
            if key in ['accuracy', 'total_cost', 'cost_per_question']:
                metadata[key] = float(value)
            elif key in ['total_questions', 'passed_questions', 'failed_questions', 'conversations_evaluated', 'random_seed']:
                metadata[key] = int(value) if value.isdigit() else value
            else:
                metadata[key] = value
    
    # Parse agent models into structured format
    if 'agent_models' in metadata:
        agent_models_str = metadata['agent_models']
        # Extract agent configurations like "supervisor(gpt-4o), extractor(gpt-4o-mini)"
        agent_configs = {}
        for match in re.finditer(r'(\w+)\(([^)]+)\)', agent_models_str):
            agent_name, model = match.groups()
            agent_configs[agent_name] = model
        metadata['agent_configs'] = agent_configs
    
    return metadata

def parse_all_results_file(results_file: Path) -> list:
    """Parse all_results.txt file to extract detailed question results."""
    results = []
    
    with open(results_file, 'r') as f:
        content = f.read()
    
    # Split by question separators
    questions = content.split('--------------------------------------------------------------------------------')
    
    for question_block in questions[1:]:  # Skip the header
        if not question_block.strip():
            continue
            
        result = {}
        lines = question_block.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('Question:'):
                result['question'] = line.replace('Question:', '').strip().strip('"')
            elif line.startswith('Expected:'):
                result['expected'] = line.replace('Expected:', '').strip()
            elif line.startswith('CrewAI Prediction:'):
                result['prediction'] = line.replace('CrewAI Prediction:', '').strip()
            elif line.startswith('Status:'):
                result['status'] = line.replace('Status:', '').strip()
                result['correct'] = '✓ CORRECT' in result['status']
            elif line.startswith('Agent Flow:'):
                result['agent_flow'] = line.replace('Agent Flow:', '').strip()
            elif line.startswith('Execution Time:'):
                time_str = line.replace('Execution Time:', '').strip().replace('s', '')
                result['execution_time'] = float(time_str)
            elif line.startswith('Estimated Cost:'):
                cost_str = line.replace('Estimated Cost:', '').strip().replace('$', '')
                result['cost'] = float(cost_str)
            elif line.startswith('Configuration:'):
                result['configuration'] = line.replace('Configuration:', '').strip()
            elif line.startswith('Operation:'):
                result['operation'] = line.replace('Operation:', '').strip()
        
        if result:
            results.append(result)
    
    return results

def main():
    """Main dashboard application."""
    
    st.title("🧪 ConvFinQA Multi-Agent Experiment Dashboard")
    st.markdown("Interactive timeline view of experiment configurations and performance")
    
    # Load data
    try:
        df = load_experiment_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    if df.empty:
        st.warning("No experiment data found in experiment_tracking/results/")
        st.info("Make sure you have run experiments using benchmark_multi_agent.py")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Date range filter
    if not df.empty and 'timestamp' in df.columns:
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Filter by date range
        if len(date_range) == 2:
            mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])
            df = df[mask]
    
    # Model filters
    if not df.empty and 'agent_configs' in df.columns:
        all_models = set()
        for configs in df['agent_configs'].dropna():
            if isinstance(configs, dict):
                all_models.update(configs.values())
        
        if all_models:
            selected_models = st.sidebar.multiselect(
                "Agent Models",
                list(all_models),
                default=list(all_models)
            )
            
            if selected_models:
                mask = df['agent_configs'].apply(
                    lambda x: any(model in x.values() for model in selected_models) if isinstance(x, dict) else False
                )
                df = df[mask]
    
    # Performance filters
    if not df.empty and 'accuracy' in df.columns:
        min_accuracy = st.sidebar.slider(
            "Minimum Accuracy (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0
        )
        df = df[df['accuracy'].fillna(0) >= min_accuracy]
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Timeline View", 
        "📊 Performance Analysis", 
        "🔧 Configuration Details",
        "📋 Detailed Results"
    ])
    
    with tab1:
        timeline_tab(df)
    
    with tab2:
        performance_tab(df)
    
    with tab3:
        configuration_tab(df)
    
    with tab4:
        detailed_results_tab(df)

def timeline_tab(df):
    """Interactive timeline view with clickable points."""
    st.header("📈 Experiment Timeline")
    
    if df.empty:
        st.info("No experiment data to display")
        return
    
    # Create the main timeline plot
    fig = go.Figure()
    
    # Add scatter plot points
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['accuracy'],
        mode='markers+lines',
        marker=dict(
            size=12,
            color=df['accuracy'],
            colorscale='RdYlGn',
            colorbar=dict(title="Accuracy (%)"),
            line=dict(width=2, color='black')
        ),
        text=df.apply(lambda row: f"Experiment: {row['experiment_id']}<br>" +
                                  f"Accuracy: {row['accuracy']:.1f}%<br>" +
                                  f"Cost: ${row['total_cost']:.3f}<br>" +
                                  f"Questions: {row['total_questions']}", axis=1),
        hovertemplate='%{text}<extra></extra>',
        name='Experiments'
    ))
    
    fig.update_layout(
        title="Experiment Performance Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Accuracy (%)",
        hovermode='closest',
        height=500
    )
    
    # Display the plot
    selected_point = st.plotly_chart(fig, use_container_width=True, key="timeline")
    
    # Experiment selection for detailed view
    st.subheader("🔍 Experiment Details")
    
    selected_exp = st.selectbox(
        "Select experiment for detailed view:",
        options=df['experiment_id'].tolist(),
        format_func=lambda x: f"{x} - {df[df['experiment_id']==x]['accuracy'].iloc[0]:.1f}%"
    )
    
    if selected_exp:
        exp_data = df[df['experiment_id'] == selected_exp].iloc[0]
        show_experiment_details(exp_data)

def show_experiment_details(exp_data):
    """Show detailed information about a selected experiment."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Performance Metrics")
        st.metric("Accuracy", f"{exp_data['accuracy']:.1f}%")
        st.metric("Total Cost", f"${exp_data['total_cost']:.3f}")
        st.metric("Questions", f"{exp_data['passed_questions']}/{exp_data['total_questions']}")
        st.metric("Cost/Question", f"${exp_data['cost_per_question']:.4f}")
    
    with col2:
        st.subheader("⚙️ Configuration")
        st.write(f"**Experiment ID:** {exp_data['experiment_id']}")
        st.write(f"**Config Hash:** {exp_data.get('config_hash', 'N/A')}")
        st.write(f"**Sampling:** {exp_data.get('sampling_strategy', 'N/A')}")
        st.write(f"**Random Seed:** {exp_data.get('random_seed', 'N/A')}")
        
        if 'agent_configs' in exp_data and isinstance(exp_data['agent_configs'], dict):
            st.write("**Agent Models:**")
            for agent, model in exp_data['agent_configs'].items():
                st.write(f"  - {agent}: {model}")
    
    # Show sample questions if available
    if 'detailed_results' in exp_data and exp_data['detailed_results']:
        st.subheader("📝 Sample Questions")
        
        results = exp_data['detailed_results']
        
        # Show performance breakdown
        correct_count = sum(1 for r in results if r.get('correct', False))
        total_count = len(results)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Correct", correct_count)
        with col2:
            st.metric("Incorrect", total_count - correct_count)
        with col3:
            st.metric("Accuracy", f"{correct_count/total_count*100:.1f}%" if total_count > 0 else "0%")
        
        # Show first few questions
        st.write("**First 5 Questions:**")
        for i, result in enumerate(results[:5]):
            status_icon = "✅" if result.get('correct', False) else "❌"
            with st.expander(f"{status_icon} Question {i+1}: {result.get('question', 'N/A')[:100]}..."):
                st.write(f"**Expected:** {result.get('expected', 'N/A')}")
                st.write(f"**Predicted:** {result.get('prediction', 'N/A')}")
                st.write(f"**Agent Flow:** {result.get('agent_flow', 'N/A')}")
                st.write(f"**Execution Time:** {result.get('execution_time', 'N/A')}s")
                st.write(f"**Cost:** ${result.get('cost', 'N/A')}")
                st.write(f"**Operation:** {result.get('operation', 'N/A')}")

def performance_tab(df):
    """Performance analysis tab."""
    st.header("📊 Performance Analysis")
    
    if df.empty:
        st.info("No data available for analysis")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Experiments", len(df))
    
    with col2:
        avg_accuracy = df['accuracy'].mean()
        st.metric("Average Accuracy", f"{avg_accuracy:.1f}%")
    
    with col3:
        total_cost = df['total_cost'].sum()
        st.metric("Total Cost", f"${total_cost:.2f}")
    
    with col4:
        total_questions = df['total_questions'].sum()
        st.metric("Total Questions", total_questions)
    
    # Performance trends
    st.subheader("📈 Performance Trends")
    
    # Accuracy over time
    fig_acc = px.line(df, x='timestamp', y='accuracy', 
                      title='Accuracy Trend Over Time',
                      labels={'accuracy': 'Accuracy (%)', 'timestamp': 'Time'})
    fig_acc.add_hline(y=df['accuracy'].mean(), line_dash="dash", 
                      annotation_text=f"Average: {df['accuracy'].mean():.1f}%")
    st.plotly_chart(fig_acc, use_container_width=True)
    
    # Cost analysis
    fig_cost = px.scatter(df, x='total_questions', y='total_cost', 
                          color='accuracy', size='total_cost',
                          title='Cost vs Questions (colored by accuracy)',
                          labels={'total_cost': 'Total Cost ($)', 'total_questions': 'Total Questions'})
    st.plotly_chart(fig_cost, use_container_width=True)

def configuration_tab(df):
    """Configuration analysis tab."""
    st.header("🔧 Configuration Analysis")
    
    if df.empty:
        st.info("No configuration data available")
        return
    
    # Model usage analysis
    if 'agent_configs' in df.columns:
        st.subheader("🤖 Agent Model Usage")
        
        model_usage = {}
        for configs in df['agent_configs'].dropna():
            if isinstance(configs, dict):
                for agent, model in configs.items():
                    key = f"{agent}:{model}"
                    model_usage[key] = model_usage.get(key, 0) + 1
        
        if model_usage:
            model_df = pd.DataFrame(list(model_usage.items()), columns=['Agent:Model', 'Count'])
            fig = px.bar(model_df, x='Agent:Model', y='Count', 
                        title='Agent Model Usage Frequency')
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Configuration performance comparison
    st.subheader("⚖️ Configuration Performance Comparison")
    
    if 'config_hash' in df.columns:
        config_perf = df.groupby('config_hash').agg({
            'accuracy': ['mean', 'std', 'count'],
            'total_cost': 'mean',
            'experiment_id': 'first'
        }).round(2)
        
        config_perf.columns = ['Avg_Accuracy', 'Std_Accuracy', 'Runs', 'Avg_Cost', 'Example_Exp']
        config_perf = config_perf.sort_values('Avg_Accuracy', ascending=False)
        
        st.dataframe(config_perf, use_container_width=True)

def detailed_results_tab(df):
    """Detailed results tab."""
    st.header("📋 Detailed Results")
    
    if df.empty:
        st.info("No detailed results available")
        return
    
    # Experiment selector
    selected_experiments = st.multiselect(
        "Select experiments to view:",
        df['experiment_id'].tolist(),
        default=df['experiment_id'].tolist()[:3]  # Default to first 3
    )
    
    if not selected_experiments:
        st.info("Please select at least one experiment")
        return
    
    # Display results for selected experiments
    for exp_id in selected_experiments:
        exp_data = df[df['experiment_id'] == exp_id].iloc[0]
        
        with st.expander(f"📊 {exp_id} - {exp_data['accuracy']:.1f}% accuracy", expanded=False):
            if 'detailed_results' in exp_data and exp_data['detailed_results']:
                results_df = pd.DataFrame(exp_data['detailed_results'])
                
                # Add summary
                correct_count = results_df['correct'].sum() if 'correct' in results_df.columns else 0
                total_count = len(results_df)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Questions", total_count)
                with col2:
                    st.metric("Correct", correct_count)
                with col3:
                    st.metric("Accuracy", f"{correct_count/total_count*100:.1f}%" if total_count > 0 else "0%")
                
                # Show results table
                display_cols = ['question', 'expected', 'prediction', 'correct', 'execution_time', 'cost']
                available_cols = [col for col in display_cols if col in results_df.columns]
                
                if available_cols:
                    st.dataframe(results_df[available_cols], use_container_width=True)
            else:
                st.info("No detailed results available for this experiment")


if __name__ == "__main__":
    main() 