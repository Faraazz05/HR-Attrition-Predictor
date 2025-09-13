"""
HR Attrition Predictor - Main Streamlit Application
==================================================
Enterprise-grade main application router with navigation, theming,
session management, and comprehensive error handling.

Author: HR Analytics Team
Date: September 2025
"""

import streamlit as st
import sys
import os
from pathlib import Path
import traceback
import logging
from datetime import datetime
import gc

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import configuration and theming
from streamlit_app.config import (
    configure_streamlit_page, initialize_session_state, 
    validate_model_files, get_app_info, NAVIGATION_PAGES,
    PAGE_CONFIG, UI_CONFIG
)
from streamlit_app.assets.theme import apply_custom_css, COLORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================================================
# PAGE IMPORTS WITH ERROR HANDLING
# ================================================================

def safe_import_pages():
    """
    Safely import all page modules with comprehensive error handling.
    
    Returns:
        Dictionary of successfully imported pages
    """
    pages = {}
    page_files = {
        "🏠 Dashboard": "pages.dashboard",
        "📊 Analytics": "pages.analytics", 
        "🔍 Predictions": "pages.predictions",
        "👥 Employee Management": "pages.employee_mgmt",
        "💡 Insights": "pages.insights",
        "⚙️ Admin": "pages.admin"
    }
    
    for page_name, module_path in page_files.items():
        try:
            # Dynamic import with error handling
            module = __import__(module_path, fromlist=[''])
            pages[page_name] = module
            logger.info(f"✅ Successfully imported: {page_name}")
        except ImportError as e:
            logger.error(f"❌ Failed to import {page_name}: {e}")
            # Create dummy page for missing modules
            pages[page_name] = create_dummy_page(page_name, str(e))
        except Exception as e:
            logger.error(f"❌ Unexpected error importing {page_name}: {e}")
            pages[page_name] = create_dummy_page(page_name, str(e))
    
    return pages

def create_dummy_page(page_name: str, error_msg: str):
    """
    Create a dummy page module for failed imports.
    
    Args:
        page_name: Name of the page
        error_msg: Error message to display
        
    Returns:
        Dummy module object
    """
    class DummyPage:
        @staticmethod
        def show():
            st.error(f"🚨 Page Loading Error: {page_name}")
            st.code(f"Error: {error_msg}")
            st.info("💡 This page is under development or missing dependencies.")
            
            with st.expander("🔧 Troubleshooting"):
                st.markdown("""
                **Possible solutions:**
                1. Check if the page file exists in the `pages/` directory
                2. Verify all required imports are available
                3. Review the error message above for specific issues
                4. Restart the application after fixing imports
                """)
    
    return DummyPage()

# ================================================================
# SYSTEM HEALTH CHECK
# ================================================================

def perform_system_health_check():
    """
    Perform comprehensive system health check on startup.
    
    Returns:
        Dictionary containing health check results
    """
    health_status = {
        'models': {},
        'data': {},
        'dependencies': {},
        'memory': {},
        'overall_status': 'healthy'
    }
    
    try:
        # Check model files
        model_status = validate_model_files()
        health_status['models'] = model_status
        
        missing_models = [k for k, v in model_status.items() if not v]
        if missing_models:
            health_status['overall_status'] = 'warning'
            logger.warning(f"Missing model files: {missing_models}")
        
        # Check data directory
        data_dirs = ['data', 'data/synthetic', 'models', 'reports', 'logs']
        for dir_path in data_dirs:
            exists = os.path.exists(dir_path)
            health_status['data'][dir_path] = exists
            if not exists:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
        
        # Check memory usage
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        health_status['memory'] = {
            'current_mb': memory_mb,
            'status': 'ok' if memory_mb < 3000 else 'high'
        }
        
        # Check critical dependencies
        critical_deps = ['pandas', 'numpy', 'plotly', 'sklearn']
        for dep in critical_deps:
            try:
                __import__(dep)
                health_status['dependencies'][dep] = True
            except ImportError:
                health_status['dependencies'][dep] = False
                health_status['overall_status'] = 'error'
        
        logger.info(f"System health check completed: {health_status['overall_status']}")
        
    except Exception as e:
        health_status['overall_status'] = 'error'
        logger.error(f"Health check failed: {e}")
    
    return health_status

# ================================================================
# SESSION STATE MANAGEMENT
# ================================================================

def setup_session_state():
    """Setup and manage Streamlit session state."""
    
    # Initialize session state
    initialize_session_state()
    
    # Add application-specific state variables
    app_state_defaults = {
        'app_initialized': False,
        'health_check_done': False,
        'models_loaded': False,
        'current_user_role': 'viewer',
        'theme_applied': False,
        'error_count': 0,
        'last_activity': datetime.now(),
        'page_load_times': {},
        'memory_warnings': 0
    }
    
    for key, default_value in app_state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def monitor_memory_usage():
    """Monitor and alert on high memory usage."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Warning threshold for 4GB systems
        if memory_mb > 3000:  # 3GB threshold
            st.session_state.memory_warnings += 1
            if st.session_state.memory_warnings % 5 == 0:  # Every 5th warning
                st.warning(f"⚠️ High memory usage: {memory_mb:.0f}MB. Consider refreshing the page.")
                
                # Force garbage collection
                gc.collect()
        
        # Update session state
        st.session_state.current_memory_mb = memory_mb
        
    except ImportError:
        pass  # psutil not available

# ================================================================
# ERROR HANDLING AND LOGGING
# ================================================================

def handle_application_error(error: Exception, context: str = ""):
    """
    Comprehensive error handling with user-friendly messages.
    
    Args:
        error: Exception that occurred
        context: Context where the error occurred
    """
    error_msg = str(error)
    error_type = type(error).__name__
    
    # Log the error
    logger.error(f"Application error in {context}: {error_type} - {error_msg}")
    logger.error(traceback.format_exc())
    
    # Update error count
    st.session_state.error_count += 1
    
    # Display user-friendly error message
    st.error(f"🚨 Application Error")
    
    with st.expander("🔍 Error Details", expanded=False):
        st.code(f"""
Error Type: {error_type}
Context: {context}
Message: {error_msg}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session Errors: {st.session_state.error_count}
        """)
        
        st.markdown("""
        **Recommended Actions:**
        1. 🔄 Refresh the page to reset the application
        2. 🧹 Clear browser cache if issues persist  
        3. 📱 Try using a different browser
        4. 💾 Ensure sufficient system memory is available
        5. 📞 Contact support if errors continue
        """)
    
    # Auto-recovery suggestions
    if "memory" in error_msg.lower():
        st.info("💡 This appears to be a memory-related issue. Try refreshing the page.")
    elif "model" in error_msg.lower():
        st.info("💡 This appears to be a model loading issue. Check if model files exist.")
    elif "import" in error_msg.lower():
        st.info("💡 This appears to be a dependency issue. Check if required packages are installed.")

# ================================================================
# NAVIGATION SETUP
# ================================================================

def create_navigation_pages():
    """
    Create navigation pages with error handling and health checks.
    
    Returns:
        Dictionary of navigation pages
    """
    # Import pages safely
    pages = safe_import_pages()
    
    # Verify page functions exist
    verified_pages = {}
    for page_name, page_module in pages.items():
        if hasattr(page_module, 'show') and callable(page_module.show):
            verified_pages[page_name] = page_module.show
        else:
            logger.warning(f"Page {page_name} missing 'show' function")
            # Create wrapper function
            def dummy_show():
                st.error(f"🚨 {page_name} is not properly configured")
                st.info("Page is missing the required 'show()' function.")
            
            verified_pages[page_name] = dummy_show
    
    return verified_pages

# ================================================================
# APPLICATION HEADER
# ================================================================

def render_application_header():
    """Render the application header with branding and status."""
    
    # Main header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="main-header">
            🏢 HR Attrition Predictor
        </h1>
        <p style="color: #B8C5D1; font-size: 1.1rem; margin-top: -10px;">
            AI-Powered Employee Retention Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status indicators
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        memory_status = "🟢" if st.session_state.get('current_memory_mb', 0) < 2000 else "🟡"
        st.caption(f"{memory_status} Memory: {st.session_state.get('current_memory_mb', 0):.0f}MB")
    
    with col2:
        model_status = "🟢" if st.session_state.get('models_loaded', False) else "🔴"
        st.caption(f"{model_status} Models")
    
    with col3:
        health_status = "🟢" if st.session_state.get('health_check_done', False) else "🟡"
        st.caption(f"{health_status} System")
    
    with col4:
        user_role = st.session_state.get('current_user_role', 'viewer').title()
        st.caption(f"👤 Role: {user_role}")
    
    with col5:
        st.caption(f"🕐 {datetime.now().strftime('%H:%M')}")
    
    st.markdown("---")

# ================================================================
# MAIN APPLICATION FUNCTION
# ================================================================

def main():
    """
    Main application function with comprehensive initialization and error handling.
    """
    
    try:
        # 1. Configure Streamlit page (must be first)
        configure_streamlit_page()
        
        # 2. Setup session state
        setup_session_state()
        
        # 3. Apply custom CSS theme
        if not st.session_state.theme_applied:
            apply_custom_css()
            st.session_state.theme_applied = True
        
        # 4. Perform system health check (only once per session)
        if not st.session_state.health_check_done:
            with st.spinner("🔍 Performing system health check..."):
                health_status = perform_system_health_check()
                st.session_state.health_check_done = True
                
                # Display health status
                if health_status['overall_status'] == 'error':
                    st.error("🚨 System health check failed. Some features may not work properly.")
                elif health_status['overall_status'] == 'warning':
                    st.warning("⚠️ System health check found some issues. Check logs for details.")
        
        # 5. Monitor memory usage
        monitor_memory_usage()
        
        # 6. Render application header
        render_application_header()
        
        # 7. Create navigation pages
        pages = create_navigation_pages()
        
        # 8. Setup navigation with error handling
        try:
            # Create navigation using Streamlit's built-in navigation
            pg = st.navigation(pages)
            
            # Track page load time
            start_time = datetime.now()
            
            # Run the selected page
            pg.run()
            
            # Record page load time
            load_time = (datetime.now() - start_time).total_seconds()
            current_page = st.session_state.get('current_page', 'Unknown')
            st.session_state.page_load_times[current_page] = load_time
            
            # Update last activity
            st.session_state.last_activity = datetime.now()
            
        except Exception as nav_error:
            handle_application_error(nav_error, "Navigation")
            
            # Fallback to manual navigation
            st.error("Navigation system failed. Using fallback mode.")
            
            # Simple selectbox navigation as fallback
            page_options = list(pages.keys())
            selected_page = st.selectbox("📄 Select Page:", page_options)
            
            if selected_page and selected_page in pages:
                try:
                    pages[selected_page]()
                except Exception as page_error:
                    handle_application_error(page_error, f"Page: {selected_page}")
        
        # 9. Footer with app information
        render_application_footer()
        
        # 10. Mark app as fully initialized
        if not st.session_state.app_initialized:
            st.session_state.app_initialized = True
            logger.info("Application fully initialized successfully")
    
    except Exception as main_error:
        handle_application_error(main_error, "Main Application")
        
        # Emergency fallback
        st.markdown("""
        ## 🚨 Application Startup Failed
        
        The HR Attrition Predictor encountered a critical error during startup.
        
        **Emergency Actions:**
        1. Refresh the browser page
        2. Clear browser cache and cookies
        3. Check system memory availability
        4. Verify all required files are present
        5. Contact technical support
        
        **System Information:**
        - Time: {time}
        - Memory: {memory} MB
        - Error Count: {errors}
        """.format(
            time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            memory=st.session_state.get('current_memory_mb', 0),
            errors=st.session_state.get('error_count', 0)
        ))

# ================================================================
# APPLICATION FOOTER
# ================================================================

def render_application_footer():
    """Render application footer with system information."""
    
    st.markdown("---")
    
    # Footer information
    footer_cols = st.columns([2, 1, 1, 1, 1])
    
    with footer_cols[0]:
        app_info = get_app_info()
        st.caption(f"🏢 {app_info['name']} v{app_info['version']} | Built with {app_info['framework']}")
    
    with footer_cols[1]:
        if st.session_state.get('page_load_times'):
            avg_load_time = sum(st.session_state.page_load_times.values()) / len(st.session_state.page_load_times)
            st.caption(f"⚡ Avg Load: {avg_load_time:.2f}s")
        else:
            st.caption("⚡ Load Time: --")
    
    with footer_cols[2]:
        uptime = datetime.now() - st.session_state.get('session_start', datetime.now())
        uptime_minutes = int(uptime.total_seconds() / 60)
        st.caption(f"⏱️ Uptime: {uptime_minutes}m")
    
    with footer_cols[3]:
        error_count = st.session_state.get('error_count', 0)
        status_emoji = "🟢" if error_count == 0 else "🟡" if error_count < 5 else "🔴"
        st.caption(f"{status_emoji} Errors: {error_count}")
    
    with footer_cols[4]:
        st.caption(f"📊 Theme: Cyberpunk")
    
    # Debug information (only show if there are issues)
    if st.session_state.get('error_count', 0) > 0 or st.session_state.get('memory_warnings', 0) > 0:
        with st.expander("🔧 System Diagnostics", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.json({
                    "Memory Usage (MB)": st.session_state.get('current_memory_mb', 0),
                    "Memory Warnings": st.session_state.get('memory_warnings', 0),
                    "Error Count": st.session_state.get('error_count', 0),
                    "Models Loaded": st.session_state.get('models_loaded', False)
                })
            
            with col2:
                if st.session_state.get('page_load_times'):
                    st.json(st.session_state.page_load_times)
                
                # Memory cleanup button
                if st.button("🧹 Force Memory Cleanup"):
                    gc.collect()
                    st.success("Memory cleanup performed")
                    st.experimental_rerun()

# ================================================================
# APPLICATION ENTRY POINT
# ================================================================

if __name__ == "__main__":
    try:
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Start the application
        logger.info("🚀 Starting HR Attrition Predictor Application")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        main()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as startup_error:
        logger.critical(f"Critical startup error: {startup_error}")
        print(f"🚨 Critical Error: {startup_error}")
        print("Please check logs for detailed error information.")
