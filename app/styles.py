"""
app/styles.py

Custom CSS styling and design system for ChurnGuard AI Streamlit Dashboard.
Provides responsive card layouts, metric badges, risk status colors, and typography.
"""


def load_custom_css():
    """Returns custom CSS markup for Streamlit app."""
    return """
    <style>
    /* Main container padding */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Metric card styling */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }

    .metric-value-high {
        font-size: 1.8rem;
        font-weight: 700;
        color: #dc3545;
    }

    .metric-value-med {
        font-size: 1.8rem;
        font-weight: 700;
        color: #fd7e14;
    }

    .metric-value-low {
        font-size: 1.8rem;
        font-weight: 700;
        color: #198754;
    }

    .metric-value-neutral {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0d6efd;
    }

    /* Risk badges */
    .badge-high {
        background-color: #f8d7da;
        color: #842029;
        border: 1px solid #f5c2c7;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }

    .badge-med {
        background-color: #fff3cd;
        color: #664d03;
        border: 1px solid #ffecb5;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }

    .badge-low {
        background-color: #d1e7dd;
        color: #0f5132;
        border: 1px solid #badbcc;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }

    /* Section header styling */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #212529;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.4rem;
    }

    /* Recommendation card */
    .recommendation-card {
        background-color: #f0f7ff;
        border-left: 4px solid #0d6efd;
        border-radius: 4px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.95rem;
        color: #1e293b;
    }

    /* Non-causal note */
    .non-causal-note {
        font-size: 0.8rem;
        color: #6c757d;
        font-style: italic;
        margin-top: 0.5rem;
    }
    </style>
    """
