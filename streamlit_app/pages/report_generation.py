# streamlit_app/pages/report_generation.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import base64
from datetime import datetime
import io
from edu_analytics.utils import save_file, get_timestamped_filename

# Add the parent directory to path if running this file directly
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from edu_analytics.utils import create_report_header

def show_report_generation():
    st.markdown("<div class='subheader'>Report Generation</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-text'>Generate a comprehensive HTML report of your analysis.</div>", unsafe_allow_html=True)
    
    # Check if we have enough data for a report
    if 'report_data' not in st.session_state:
        st.warning("No analysis data available for report generation. Please perform some analysis first.")
        return
    
    # Debug: Show stored visualizations
    debug_expander = st.expander("Debug: View stored visualizations", expanded=False)
    with debug_expander:
        for category in ['exploration', 'statistical', 'model', 'prediction']:
            key = f"{category}_figures"
            if key in st.session_state:
                st.write(f"**{category.capitalize()} Figures:** {len(st.session_state[key])}")
                for i, (title, _) in enumerate(st.session_state[key]):
                    st.write(f"  {i+1}. {title}")
            else:
                st.write(f"**{category.capitalize()} Figures:** None")
    
    # Report options
    st.markdown("### Report Options")
    
    # Get available sections
    available_sections = []
    if 'data' in st.session_state and st.session_state.data is not None:
        available_sections.append("Dataset Overview")
    if 'statistical_tests' in st.session_state.report_data and st.session_state.report_data['statistical_tests']:
        available_sections.append("Statistical Analysis")
    if 'threshold_analysis' in st.session_state.report_data and st.session_state.report_data['threshold_analysis']:
        available_sections.append("Threshold Analysis")
    if 'model_training' in st.session_state.report_data and st.session_state.report_data['model_training']:
        available_sections.append("Model Training")
    if 'model_evaluation' in st.session_state.report_data and st.session_state.report_data['model_evaluation']:
        available_sections.append("Model Evaluation")
    if ('predictions' in st.session_state.report_data and st.session_state.report_data['predictions']) or \
       ('batch_predictions' in st.session_state.report_data and st.session_state.report_data['batch_predictions']):
        available_sections.append("Predictions")
    
    # Allow user to select sections to include
    selected_sections = st.multiselect(
        "Select sections to include in the report",
        available_sections,
        default=available_sections
    )
    
    # Initialize or clear selected visualizations
    if 'selected_visualizations' not in st.session_state:
        st.session_state.selected_visualizations = []
    
    # Visualizations section
    st.markdown("### Include Visualizations")
    st.info("Select visualizations to include in your report")
    
    # Create a tabbed interface for visualizations
    visualization_tabs = st.tabs(["Data Exploration", "Statistical Tests", "Models", "Predictions"])
    
    # Helper function to display figures with checkboxes
    def display_figures(figures, prefix):
        if not figures:
            st.info(f"No visualizations available in this category.")
            return
            
        for i, (title, fig) in enumerate(figures):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.pyplot(fig)
            with col2:
                include_viz = st.checkbox(f"Include in report", key=f"{prefix}_{i}")
                if include_viz:
                    # Add to selected visualizations if not already there
                    if (title, fig) not in st.session_state.selected_visualizations:
                        st.session_state.selected_visualizations.append((title, fig))
                else:
                    # Remove if it was previously selected
                    st.session_state.selected_visualizations = [
                        (t, f) for t, f in st.session_state.selected_visualizations 
                        if t != title
                    ]
    
    # Data Exploration visualizations
    with visualization_tabs[0]:
        st.subheader("Data Exploration Visualizations")
        if 'exploration_figures' in st.session_state:
            display_figures(st.session_state.exploration_figures, "viz_exp")
        else:
            st.info("No data exploration visualizations available.")
    
    # Statistical test visualizations
    with visualization_tabs[1]:
        st.subheader("Statistical Test Visualizations")
        if 'statistical_figures' in st.session_state:
            display_figures(st.session_state.statistical_figures, "viz_stat")
        else:
            st.info("No statistical test visualizations available.")
    
    # Model visualizations
    with visualization_tabs[2]:
        st.subheader("Model Visualizations")
        if 'model_figures' in st.session_state:
            display_figures(st.session_state.model_figures, "viz_model")
        else:
            st.info("No model visualizations available.")
    
    # Prediction visualizations
    with visualization_tabs[3]:
        st.subheader("Prediction Visualizations")
        if 'prediction_figures' in st.session_state:
            display_figures(st.session_state.prediction_figures, "viz_pred")
        else:
            st.info("No prediction visualizations available.")
    
    # Display summary of selected visualizations
    if st.session_state.selected_visualizations:
        st.success(f"{len(st.session_state.selected_visualizations)} visualizations selected for the report")
    else:
        st.info("No visualizations selected for the report")
    
    # Report metadata
    st.markdown("### Report Metadata")
    report_title = st.text_input("Report Title", value="Statistical Analysis Report")
    author = st.text_input("Author", value="")
    
    # Generate report button
    if st.button("Generate Report", key="gen_report"):
        if not selected_sections:
            st.warning("Please select at least one section to include in the report.")
            return
        try:
            with st.spinner("Generating report..."):
                # Create report content
                report_html = generate_html_report(
                    title=report_title,
                    author=author,
                    sections=selected_sections,
                    report_data=st.session_state.report_data,
                    data=st.session_state.data if 'data' in st.session_state else None,
                    target_type=st.session_state.target_type if 'target_type' in st.session_state else None,
                    target_column=st.session_state.processed_data['target_column'] if 'processed_data' in st.session_state else None,
                    visualizations=st.session_state.selected_visualizations
                )
                
                # Prepare the HTML report
                html_report = prepare_report(report_html)
                
                # Generate filename
                html_filename = get_timestamped_filename("statistical_analysis_report", "html")
                
                # Save HTML report
                html_success, html_message, html_path = save_file(html_report, html_filename, "reports")
                
                # Show success/error messages
                if html_success:
                    st.success(f"HTML report saved: {html_path}")
                else:
                    st.warning(html_message)
                
                # Provide download button
                st.download_button(
                    label="Download Report (HTML)",
                    data=html_report,
                    file_name=html_filename,
                    mime='text/html',
                )
                
                # Show preview
                st.markdown("### Report Preview")
                st.components.v1.html(report_html, height=600, scrolling=True)
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def figure_to_base64(fig):
    """
    Convert a matplotlib figure to base64 encoded string
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to convert
    
    Returns:
    --------
    str : Base64 encoded image string
    """
    from io import BytesIO
    import base64
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Helper function to convert HTML to PDF
def prepare_report(html_content):
    """
    Prepare the HTML report for download
    Parameters:
    -----------
    html_content : str
        HTML content of the report
    Returns:
    --------
    bytes : HTML content as bytes
    """
    # Add any necessary processing to the HTML content
    enhanced_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Statistical Analysis Report</title>
        <style>
            @media print {{
                @page {{ size: letter; margin: 1cm; }}
                body {{ font-size: 12px; }}
                .no-print {{ display: none; }}
            }}
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3, h4 {{ color: #1E88E5; }}
            img {{ max-width: 100%; height: auto; }}
            table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; }}
            th {{ background-color: #f2f2f2; }}
            .section {{ margin-top: 2rem; border-top: 1px solid #eee; padding-top: 1rem; }}
            .footer {{ margin-top: 2rem; text-align: center; font-size: 0.9rem; color: #666; }}
            .chart-container {{ 
                text-align: center; 
                margin: 1rem auto;
                page-break-inside: avoid;
            }}
            /* Add print button styles */
            .print-button {{
                background-color: #1E88E5;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                cursor: pointer;
                margin: 1rem 0;
                font-weight: bold;
            }}
            .print-button:hover {{
                background-color: #1976D2;
            }}
        </style>
    </head>
    <body>
        <!-- Add print button -->
        <div class="no-print" style="text-align: right;">
            <button class="print-button" onclick="window.print()">Print Report</button>
        </div>
        
        {html_content}
        
        <script>
            // Add any JavaScript functionality here
            document.addEventListener('DOMContentLoaded', function() {{
                // Auto-resize tables if they're too wide
                const tables = document.querySelectorAll('table');
                tables.forEach(table => {{
                    if (table.offsetWidth > table.parentElement.offsetWidth) {{
                        table.style.fontSize = '0.9rem';
                    }}
                }});
            }});
        </script>
    </body>
    </html>
    """
    return enhanced_html.encode('utf-8')
        
# Helper function to generate HTML report
def generate_html_report(
    title, 
    author, 
    sections, 
    report_data, 
    data=None, 
    target_type=None, 
    target_column=None, 
    visualizations=None
):
    """
    Generate an HTML report with the given sections and data
    Parameters:
    -----------
    title : str
        Report title
    author : str
        Report author
    sections : list
        List of sections to include
    report_data : dict
        Report data from session state
    data : DataFrame, optional
        Original dataset
    target_type : str, optional
        Type of target variable
    target_column : str, optional
        Target column name
    visualizations : list, optional
        List of (title, figure) tuples to include in the report
    Returns:
    --------
    str : HTML report content
    """
    # Get current date and time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start HTML content
    html = f"""
    <div class="header">
        <h1>{title}</h1>
    </div>
    <div class="metadata">
        <p>Generated on: {now}</p>
        {f'<p>Author: {author}</p>' if author else ''}
    </div>
    """
    
    # Add selected visualizations section if any are provided
    if visualizations and len(visualizations) > 0:
        html += """
        <div class="section">
            <h2>Selected Visualizations</h2>
            <div class="visualization-gallery">
        """
        
        for i, (viz_title, fig) in enumerate(visualizations):
            # Convert matplotlib figure to base64 encoded PNG
            from io import BytesIO
            import base64
            
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            html += f"""
            <div class="chart-container">
                <h3>{viz_title}</h3>
                <img src="data:image/png;base64,{img_str}" alt="{viz_title}">
            </div>
            """
            
            # Add a separator unless it's the last visualization
            if i < len(visualizations) - 1:
                html += "<hr style='border-top: 1px dashed #ccc; margin: 2rem 0;'>"
        
        html += """
            </div>
        </div>
        """
    
    # Add content for each selected section
    for section in sections:
        if section == "Dataset Overview" and data is not None:
            html += generate_dataset_overview_section(data, target_type, target_column)
        elif section == "Statistical Analysis" and 'statistical_tests' in report_data:
            html += generate_statistical_analysis_section(report_data['statistical_tests'])
        elif section == "Threshold Analysis" and 'threshold_analysis' in report_data:
            html += generate_threshold_analysis_section(report_data['threshold_analysis'])
        elif section == "Model Training" and 'model_training' in report_data:
            html += generate_model_training_section(report_data['model_training'])
        elif section == "Model Evaluation" and 'model_evaluation' in report_data:
            html += generate_model_evaluation_section(report_data['model_evaluation'])
        elif section == "Predictions" and ('predictions' in report_data or 'batch_predictions' in report_data):
            html += generate_predictions_section(report_data)
    
    # Add footer
    html += f"""
    <div class="footer">
        <p>Generated using Statistical Analysis App</p>
        <p>© {datetime.now().year}</p>
    </div>
    """
    
    return html

# Helper function to generate dataset overview section
def generate_dataset_overview_section(data, target_type, target_column):
    """Generate HTML for dataset overview section"""
    
    # Basic dataset stats
    rows = data.shape[0]
    cols = data.shape[1]
    missing_values = data.isna().sum().sum()
    missing_pct = (missing_values / (rows * cols)) * 100
    
    # Column types breakdown
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    html = f"""
        <div class="section">
            <h2>Dataset Overview</h2>
            
            <h3>Basic Statistics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Number of Rows</td>
                    <td>{rows}</td>
                </tr>
                <tr>
                    <td>Number of Columns</td>
                    <td>{cols}</td>
                </tr>
                <tr>
                    <td>Missing Values</td>
                    <td>{missing_values} ({missing_pct:.2f}%)</td>
                </tr>
                <tr>
                    <td>Numeric Columns</td>
                    <td>{len(numeric_cols)}</td>
                </tr>
                <tr>
                    <td>Categorical Columns</td>
                    <td>{len(categorical_cols)}</td>
                </tr>
                <tr>
                    <td>Target Column</td>
                    <td>{target_column} ({target_type if target_type else 'Unknown'})</td>
                </tr>
            </table>
            
            <h3>Column Information</h3>
            <table>
                <tr>
                    <th>Column Name</th>
                    <th>Type</th>
                    <th>Non-Null Count</th>
                    <th>Unique Values</th>
                </tr>
    """
    
    # Add row for each column
    for col in data.columns:
        col_type = data[col].dtype
        non_null = data[col].count()
        unique = data[col].nunique()
        html += f"""
                <tr>
                    <td>{col}</td>
                    <td>{col_type}</td>
                    <td>{non_null} ({(non_null/rows)*100:.1f}%)</td>
                    <td>{unique}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
    """
    
    return html

# Helper function to generate statistical analysis section
def generate_statistical_analysis_section(statistical_tests):
    """Generate HTML for statistical analysis section"""
    
    if not statistical_tests:
        return ""
    
    html = f"""
        <div class="section">
            <h2>Statistical Analysis</h2>
            <p>This section presents the results of various statistical tests performed on the data.</p>
    """
    
    # Process each type of test
    for test_key, test_data in statistical_tests.items():
        if test_key == 'summary':
            continue
        
        test_type = test_data.get('type', 'unknown')
        description = test_data.get('description', 'Statistical Analysis')
        
        html += f"""
            <div class="subsection">
                <h3>{description}</h3>
        """
        
        if test_type == 't-test':
            results = test_data.get('results', {})
            
            html += f"""
                <p>
                    Comparing means of <span class="highlight">{results.get('feature', '')}</span> 
                    between groups of <span class="highlight">{results.get('target', '')}</span>.
                </p>
                
                <table>
                    <tr>
                        <th>Group</th>
                        <th>Mean</th>
                        <th>Standard Deviation</th>
                        <th>Count</th>
                    </tr>
                    <tr>
                        <td>{results.get('groups', ['Group 1', 'Group 2'])[0]}</td>
                        <td>{results.get('group1_mean', 0):.4f}</td>
                        <td>{results.get('group1_std', 0):.4f}</td>
                        <td>{results.get('group1_n', 0)}</td>
                    </tr>
                    <tr>
                        <td>{results.get('groups', ['Group 1', 'Group 2'])[1]}</td>
                        <td>{results.get('group2_mean', 0):.4f}</td>
                        <td>{results.get('group2_std', 0):.4f}</td>
                        <td>{results.get('group2_n', 0)}</td>
                    </tr>
                </table>
                
                <h4>Test Results</h4>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Mean Difference</td>
                        <td class="metric">{results.get('mean_difference', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td>t-statistic</td>
                        <td>{results.get('t_statistic', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td>p-value</td>
                        <td class="metric">{results.get('p_value', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td>Test Type</td>
                        <td>{results.get('test_type', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <td>Significance</td>
                        <td>{f'<span class="highlight">Significant</span>' if results.get('significant', False) else 'Not significant'}</td>
                    </tr>
                </table>
                
                <p>
                    <strong>Interpretation:</strong> 
                    {f'There is a statistically significant difference in the means of {results.get("feature", "")} between the two groups (p < {results.get("alpha", 0.05)}).' 
                    if results.get('significant', False) 
                    else f'There is no statistically significant difference in the means of {results.get("feature", "")} between the two groups (p > {results.get("alpha", 0.05)}).'}
                </p>
            """
        
        elif test_type == 'chi-square':
            results = test_data.get('results', {})
            
            html += f"""
                <p>
                    Testing association between <span class="highlight">{results.get('feature', '')}</span> 
                    and <span class="highlight">{results.get('target', '')}</span>.
                </p>
                
                <h4>Test Results</h4>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Chi-square Statistic</td>
                        <td class="metric">{results.get('chi2', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td>p-value</td>
                        <td class="metric">{results.get('p_value', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td>Degrees of Freedom</td>
                        <td>{results.get('dof', 0)}</td>
                    </tr>
                    <tr>
                        <td>Cramer's V</td>
                        <td class="metric">{results.get('cramers_v', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td>Effect Size</td>
                        <td>{results.get('effect_size', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <td>Significance</td>
                        <td>{f'<span class="highlight">Significant</span>' if results.get('significant', False) else 'Not significant'}</td>
                    </tr>
                </table>
                
                <p>
                    <strong>Interpretation:</strong> 
                    {f'There is a statistically significant association between {results.get("feature", "")} and {results.get("target", "")} (p < {results.get("alpha", 0.05)}). The effect size is {results.get("effect_size", "unknown").lower()} (Cramer\'s V = {results.get("cramers_v", 0):.4f}).' 
                    if results.get('significant', False) 
                    else f'There is no statistically significant association between {results.get("feature", "")} and {results.get("target", "")} (p > {results.get("alpha", 0.05)}).'}
                </p>
            """
        
        elif test_type == 'anova':
            results = test_data.get('results', {})
            
            html += f"""
                <p>
                    Comparing means of <span class="highlight">{results.get('feature', '')}</span> 
                    across groups of <span class="highlight">{results.get('group', '')}</span>.
                </p>
                
                <h4>Test Results</h4>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>F-statistic</td>
                        <td class="metric">{results.get('f_statistic', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td>p-value</td>
                        <td class="metric">{results.get('p_value', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td>Degrees of Freedom (between, within)</td>
                        <td>{results.get('df_between', 0)}, {results.get('df_within', 0)}</td>
                    </tr>
                    <tr>
                        <td>Eta-squared</td>
                        <td class="metric">{results.get('eta_squared', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td>Effect Size</td>
                        <td>{results.get('effect_size', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <td>Significance</td>
                        <td>{f'<span class="highlight">Significant</span>' if results.get('significant', False) else 'Not significant'}</td>
                    </tr>
                </table>
                
                <p>
                    <strong>Interpretation:</strong> 
                    {f'There are statistically significant differences in the means of {results.get("feature", "")} across {results.get("group", "")} groups (p < {results.get("alpha", 0.05)}). The effect size is {results.get("effect_size", "unknown").lower()} (Eta-squared = {results.get("eta_squared", 0):.4f}).' 
                    if results.get('significant', False) 
                    else f'There are no statistically significant differences in the means of {results.get("feature", "")} across {results.get("group", "")} groups (p > {results.get("alpha", 0.05)}).'}
                </p>
            """
        
        elif test_type == 'correlation':
            # Correlation analysis often has a summary DataFrame
            if 'summary' in test_data.get('results', {}):
                summary = test_data['results']['summary']
                
                html += """
                    <h4>Correlation Summary</h4>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Correlation</th>
                            <th>p-value</th>
                            <th>R-squared</th>
                            <th>Effect Size</th>
                            <th>Significance</th>
                        </tr>
                """
                
                # Limit to top correlations for readability
                if isinstance(summary, pd.DataFrame):
                    top_correlations = summary.head(10).to_dict('records')
                    
                    for row in top_correlations:
                        html += f"""
                            <tr>
                                <td>{row.get('feature', '')}</td>
                                <td class="metric">{row.get('correlation', 0):.4f}</td>
                                <td>{row.get('p_value', 0):.4f}</td>
                                <td>{row.get('r_squared', 0):.4f}</td>
                                <td>{row.get('effect_size', 'Unknown')}</td>
                                <td>{'<span class="highlight">Significant</span>' if row.get('significant', False) else 'Not significant'}</td>
                            </tr>
                        """
                
                html += """
                    </table>
                    <p><em>Note: Showing up to 10 top correlations.</em></p>
                """
        
        elif test_type == 'feature_importance':
            # Feature importance often has a DataFrame with features and importance scores
            results = test_data.get('results', pd.DataFrame())
            
            html += """
                <h4>Feature Importance Ranking</h4>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                    </tr>
            """
            
            # Show top features by importance
            if isinstance(results, pd.DataFrame):
                # Sort by importance and get top 10
                top_features = results.sort_values('Importance', ascending=False).head(10).to_dict('records')
                
                for row in top_features:
                    html += f"""
                        <tr>
                            <td>{row.get('Feature', '')}</td>
                            <td class="metric">{row.get('Importance', 0):.4f}</td>
                        </tr>
                    """
            
            html += """
                </table>
                <p><em>Note: Showing up to 10 top features.</em></p>
                
                <p>
                    <strong>Interpretation:</strong> 
                    The table above ranks features by their importance in predicting the target variable.
                    Features with higher importance scores have a stronger influence on the predictions.
                </p>
            """
        
        html += """
            </div>
        """
    
    html += """
        </div>
    """
    
    return html

# Helper function to generate threshold analysis section
def generate_threshold_analysis_section(threshold_analysis):
    """Generate HTML for threshold analysis section"""
    
    if not threshold_analysis:
        return ""
    
    html = f"""
        <div class="section">
            <h2>Threshold Analysis</h2>
            <p>
                This section analyzes how different feature thresholds affect the target variable, 
                helping identify important decision boundaries in your data.
            </p>
    """
    
    # Process each threshold analysis
    for analysis_key, analysis_data in threshold_analysis.items():
        analysis_type = analysis_data.get('type', 'unknown')
        
        if analysis_type == 'single_feature':
            html += f"""
                <div class="subsection">
                    <h3>Single Feature Threshold Analysis</h3>
                    <p>
                        Analysis of how different thresholds for 
                        <span class="highlight">{analysis_data.get('feature', '')}</span> 
                        affect <span class="highlight">{analysis_data.get('target', '')}</span>.
                    </p>
            """
            
            # Add detailed results if available
            if 'results' in analysis_data and 'features' in analysis_data['results']:
                feature = analysis_data.get('feature', '')
                if feature in analysis_data['results']['features']:
                    feature_results = analysis_data['results']['features'][feature]
                    
                    # Determine if this is a time variable
                    is_time = feature_results.get('is_time_variable', False)
                    
                    html += f"""
                        <h4>Optimal Threshold</h4>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Optimal Threshold</td>
                                <td class="metric">{feature_results.get('optimal_threshold_display', '')}</td>
                            </tr>
                    """
                    
                    if is_time:
                        html += f"""
                            <tr>
                                <td>Rate Below Threshold (Faster Times)</td>
                                <td class="metric">{feature_results.get('optimal_above_rate', 0):.2%}</td>
                            </tr>
                            <tr>
                                <td>Rate Above Threshold (Slower Times)</td>
                                <td class="metric">{feature_results.get('optimal_below_rate', 0):.2%}</td>
                            </tr>
                        """
                    else:
                        html += f"""
                            <tr>
                                <td>Rate Above Threshold</td>
                                <td class="metric">{feature_results.get('optimal_above_rate', 0):.2%}</td>
                            </tr>
                            <tr>
                                <td>Rate Below Threshold</td>
                                <td class="metric">{feature_results.get('optimal_below_rate', 0):.2%}</td>
                            </tr>
                        """
                    
                    html += f"""
                            <tr>
                                <td>Difference in Rates</td>
                                <td class="metric">{feature_results.get('optimal_difference', 0):.2%}</td>
                            </tr>
                        </table>
                    """
            
            html += f"""
                    <p>
                        <em>This analysis was performed on {analysis_data.get('timestamp', '')}.</em>
                    </p>
                </div>
            """
        
        elif analysis_type == 'feature_combination':
            features = analysis_data.get('features', [])
            html += f"""
                <div class="subsection">
                    <h3>Feature Combination Analysis</h3>
                    <p>
                        Analysis of how combinations of 
                        <span class="highlight">{features[0] if len(features) > 0 else ''}</span> and
                        <span class="highlight">{features[1] if len(features) > 1 else ''}</span> 
                        affect <span class="highlight">{analysis_data.get('target', '')}</span>.
                    </p>
            """
            
            # Add quadrant analysis if available
            if ('results' in analysis_data and 
                'feature_combination' in analysis_data['results'] and 
                'quadrant_results' in analysis_data['results']['feature_combination']):
                
                quadrant_results = analysis_data['results']['feature_combination']['quadrant_results']
                
                html += """
                    <h4>Quadrant Analysis</h4>
                    <table>
                        <tr>
                            <th>Quadrant</th>
                            <th>Target Rate</th>
                            <th>Sample Size</th>
                            <th>Sample %</th>
                        </tr>
                """
                
                for quadrant, results in quadrant_results.items():
                    html += f"""
                        <tr>
                            <td>{quadrant}</td>
                            <td class="metric">{results.get('target_rate', 0):.2%}</td>
                            <td>{results.get('sample_size', 0)}</td>
                            <td>{results.get('sample_pct', 0):.1%}</td>
                        </tr>
                    """
                
                html += """
                    </table>
                """
            
            html += f"""
                    <p>
                        <em>This analysis was performed on {analysis_data.get('timestamp', '')}.</em>
                    </p>
                </div>
            """
        
        elif analysis_type == 'custom_threshold':
            html += f"""
                <div class="subsection">
                    <h3>Custom Threshold Analysis</h3>
                    <p>
                        Analysis of how specific thresholds for 
                        <span class="highlight">{analysis_data.get('feature1', '')}</span> 
                        ({analysis_data.get('threshold1', '')}) and
                        <span class="highlight">{analysis_data.get('feature2', '')}</span> 
                        ({analysis_data.get('threshold2', '')}) affect 
                        <span class="highlight">{analysis_data.get('target', '')}</span>.
                    </p>
            """
            
            # If we have results, add them
            if 'results' in analysis_data:
                results = analysis_data['results']
                
                html += """
                    <h4>Quadrant Analysis</h4>
                    <table>
                        <tr>
                            <th>Condition</th>
                            <th>Count</th>
                            <th>Target Rate</th>
                            <th>% of Total</th>
                        </tr>
                """
                
                for condition, row in results.iterrows():
                    html += f"""
                        <tr>
                            <td>{condition}</td>
                            <td>{row.get('count', 0)}</td>
                            <td class="metric">{row.get('target_1_rate', 0):.2%}</td>
                            <td>{row.get('percentage_of_total', 0):.1%}</td>
                        </tr>
                    """
                
                html += """
                    </table>
                """
            
            html += f"""
                    <p>
                        <em>This analysis was performed on {analysis_data.get('timestamp', '')}.</em>
                    </p>
                </div>
            """
    
    html += """
        </div>
    """
    
    return html

# Helper function to generate model training section
def generate_model_training_section(model_training):
    """Generate HTML for model training section"""
    
    if not model_training:
        return ""
    
    models_trained = model_training.get('models_trained', [])
    evaluation_results = model_training.get('evaluation_results', pd.DataFrame())
    feature_importance = model_training.get('feature_importance', None)
    
    html = f"""
        <div class="section">
            <h2>Model Training</h2>
            
            <h3>Models Trained</h3>
            <p>
                The following models were trained:
                <ul>
    """
    
    for model in models_trained:
        html += f"""
                    <li>{model}</li>
        """
    
    html += """
                </ul>
            </p>
            
            <h3>Model Performance</h3>
    """
    
    # Add evaluation results if available
    if isinstance(evaluation_results, pd.DataFrame) and not evaluation_results.empty:
        html += """
            <table>
                <tr>
                    <th>Model</th>
        """
        
        # Add column headers
        for col in evaluation_results.columns:
            html += f"""
                    <th>{col}</th>
            """
        
        html += """
                </tr>
        """
        
        # Add rows for each model
        for idx, row in evaluation_results.iterrows():
            html += f"""
                <tr>
                    <td>{idx}</td>
            """
            
            for col in evaluation_results.columns:
                # Format metric value
                value = row[col]
                if isinstance(value, (int, float)):
                    html += f"""
                        <td class="metric">{value:.4f}</td>
                    """
                else:
                    html += f"""
                        <td>{value}</td>
                    """
            
        html += """
            </table>
        """
    
    # Add feature importance if available
    if feature_importance is not None:
        html += """
            <h3>Feature Importance</h3>
        """
        
        if isinstance(feature_importance, dict):
            # If it's a dictionary of feature importances by model
            for model_name, importance_df in feature_importance.items():
                html += f"""
                    <h4>Feature Importance from {model_name}</h4>
                """
                
                if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
                    # Sort by importance
                    sorted_importance = importance_df.sort_values('importance', ascending=False)
                    
                    html += """
                        <table>
                            <tr>
                                <th>Feature</th>
                                <th>Importance</th>
                            </tr>
                    """
                    
                    # Limit to top 15 features for readability
                    for _, row in sorted_importance.head(15).iterrows():
                        html += f"""
                            <tr>
                                <td>{row.get('feature', '')}</td>
                                <td class="metric">{row.get('importance', 0):.4f}</td>
                            </tr>
                        """
                    
                    html += """
                        </table>
                        <p><em>Note: Showing top 15 features by importance.</em></p>
                    """
        
        elif isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty:
            # Sort by importance
            sorted_importance = feature_importance.sort_values('importance', ascending=False)
            
            html += """
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                    </tr>
            """
            
            # Limit to top 15 features for readability
            for _, row in sorted_importance.head(15).iterrows():
                html += f"""
                    <tr>
                        <td>{row.get('feature', '')}</td>
                        <td class="metric">{row.get('importance', 0):.4f}</td>
                    </tr>
                """
            
            html += """
                </table>
                <p><em>Note: Showing top 15 features by importance.</em></p>
            """
    
    html += f"""
            <p>
                <em>Models were trained on {model_training.get('timestamp', '')}.</em>
            </p>
        </div>
    """
    
    return html

# Helper function to generate model evaluation section
def generate_model_evaluation_section(model_evaluation):
    """Generate HTML for model evaluation section"""
    
    if not model_evaluation:
        return ""
    
    html = f"""
        <div class="section">
            <h2>Model Evaluation</h2>
            
            <p>This section presents the evaluation results for the trained models.</p>
    """
    
    # Add details if available
    if isinstance(model_evaluation, dict) and 'models_evaluated' in model_evaluation:
        models_evaluated = model_evaluation.get('models_evaluated', [])
        cross_validation_results = model_evaluation.get('cross_validation_results', {})
        
        html += f"""
            <h3>Models Evaluated</h3>
            <p>
                The following models were evaluated:
                <ul>
        """
        
        for model in models_evaluated:
            html += f"""
                        <li>{model}</li>
            """
        
        html += """
                </ul>
            </p>
            
            <h3>Cross-Validation Results</h3>
        """
        
        # Add cross-validation results if available
        if cross_validation_results:
            html += """
                <table>
                    <tr>
                        <th>Model</th>
            """
            
            # Get column names from first model
            first_model = list(cross_validation_results.keys())[0] if cross_validation_results else None
            if first_model:
                columns = list(cross_validation_results[first_model].keys())
                
                # Add column headers
                for col in columns:
                    html += f"""
                        <th>{col}</th>
                    """
            
                html += """
                    </tr>
                """
                
                # Add rows for each model
                for model, results in cross_validation_results.items():
                    html += f"""
                        <tr>
                            <td>{model}</td>
                    """
                    
                    for col in columns:
                        html += f"""
                            <td class="metric">{results.get(col, '')}</td>
                        """
                    
                    html += """
                        </tr>
                    """
            
            html += """
                </table>
            """
    
    html += f"""
            <p>
                <em>Evaluation was performed on {model_evaluation.get('timestamp', '')}.</em>
            </p>
        </div>
    """
    
    return html

# Helper function to generate predictions section
def generate_predictions_section(report_data):
    """Generate HTML for predictions section"""
    
    html = f"""
        <div class="section">
            <h2>Predictions</h2>
            <p>This section summarizes the predictions made using the trained models.</p>
    """
    
    # Add single predictions if available
    if 'predictions' in report_data and report_data['predictions']:
        html += """
            <h3>Individual Predictions</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Prediction</th>
                    <th>Timestamp</th>
                </tr>
        """
        
        # Limit to the last 10 predictions for readability
        predictions = report_data['predictions'][-10:] if len(report_data['predictions']) > 10 else report_data['predictions']
        
        for prediction in predictions:
            html += f"""
                <tr>
                    <td>{prediction.get('model', '')}</td>
                    <td class="metric">{prediction.get('prediction', '')}</td>
                    <td>{prediction.get('timestamp', '')}</td>
                </tr>
            """
        
        html += """
            </table>
        """
        
        if len(report_data['predictions']) > 10:
            html += """
                <p><em>Note: Showing the 10 most recent predictions.</em></p>
            """
    
    # Add batch predictions if available
    if 'batch_predictions' in report_data and report_data['batch_predictions']:
        html += """
            <h3>Batch Predictions</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Records Processed</th>
                    <th>Timestamp</th>
                </tr>
        """
        
        for batch in report_data['batch_predictions']:
            html += f"""
                <tr>
                    <td>{batch.get('model', '')}</td>
                    <td>{batch.get('records_processed', 0)}</td>
                    <td>{batch.get('timestamp', '')}</td>
                </tr>
            """
        
        html += """
            </table>
        """
    
    html += """
        </div>
    """
    
    return html

# Import the file browser
from streamlit_app.components.file_browser import file_browser
import os

# Add a section to view saved reports
st.markdown("---")
st.subheader("View Saved Reports")

# Get reports directory
if 'save_directory' in st.session_state:
    reports_dir = os.path.join(st.session_state.save_directory, "reports")
    
    # Check if directory exists, create if it doesn't
    if not os.path.exists(reports_dir):
        try:
            os.makedirs(reports_dir, exist_ok=True)
            st.info(f"Created reports directory: {reports_dir}")
        except Exception as e:
            st.warning(f"Could not create reports directory: {str(e)}")
    
    # Use the file browser component
    selected_report = file_browser(reports_dir, "pdf")
    
    if selected_report:
        st.write(f"Selected report: {os.path.basename(selected_report)}")
        
        # Offer options to open the file
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Open Report"):
                try:
                    import webbrowser
                    webbrowser.open(selected_report)
                    st.success("Report opened in default application")
                except Exception as e:
                    st.error(f"Could not open report: {str(e)}")
        
        with col2:
            # Read the file and offer download
            try:
                with open(selected_report, "rb") as f:
                    report_bytes = f.read()
                    
                st.download_button(
                    "Download Report",
                    report_bytes,
                    file_name=os.path.basename(selected_report),
                    mime="application/pdf" if selected_report.endswith(".pdf") else "text/html"
                )
            except Exception as e:
                st.error(f"Could not read report file: {str(e)}")
else:
    st.info("Please set a save directory in the sidebar to view saved reports.")

if __name__ == "__main__":
    show_report_generation()