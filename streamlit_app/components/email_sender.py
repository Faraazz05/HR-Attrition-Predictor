"""
Email Sender - Frontend Email Interface
======================================
Purpose: Frontend email interface for HR notifications and reports
Author: HR Analytics Team
Date: September 2025
Version: 2.0

Functions:
- compose_email_interface(): Interactive email composition interface
- template_selector(): Email template selection and customization
- bulk_email_sender(): Bulk email operations for notifications

Connections: Uses email_service.py for backend email operations
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Optional, Union
import base64
import io
import time 

# Project imports
try:
    from src.utils.email_service import EmailService
    from  src.utils.email_service import EmailTemplates
    EMAIL_SERVICE_AVAILABLE = True
except ImportError:
    EMAIL_SERVICE_AVAILABLE = False
    print("⚠️ Email service not available - using mock implementation")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailSender:
    """Frontend email interface for HR notifications"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize email sender interface.
        
        Args:
            config_path: Path to email configuration file
        """
        self.config_path = config_path
        self.email_service = None
        self.templates = EmailTemplates() if EMAIL_SERVICE_AVAILABLE else None
        
        # Initialize email service
        if EMAIL_SERVICE_AVAILABLE:
            try:
                self.email_service = EmailService(config_path)
                logger.info("Email service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize email service: {e}")
                self.email_service = None
        
        # Email composition state
        self.composition_state = {
            'recipients': [],
            'subject': '',
            'body': '',
            'template': None,
            'attachments': [],
            'priority': 'normal',
            'send_later': False,
            'scheduled_time': None
        }
        
        # Bulk email state
        self.bulk_state = {
            'employee_list': None,
            'template_selected': None,
            'personalization_fields': {},
            'batch_size': 50,
            'send_interval': 5  # seconds between batches
        }

    def compose_email_interface(self) -> Dict:
        """
        Interactive email composition interface using Streamlit.
        
        Returns:
            dict: Composition result and status
        """
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h2 style='color: white; margin: 0; text-align: center;'>
                📧 HR Email Composer
            </h2>
            <p style='color: white; margin: 0.5rem 0 0 0; text-align: center; opacity: 0.9;'>
                Send targeted HR communications and attrition alerts
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Email composition tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📝 Compose", "📋 Templates", "👥 Recipients", "⚙️ Settings"
        ])
        
        with tab1:
            result = self._compose_tab()
        
        with tab2:
            self._templates_tab()
        
        with tab3:
            self._recipients_tab()
        
        with tab4:
            self._settings_tab()
        
        # Send button in sidebar
        with st.sidebar:
            st.markdown("### 🚀 Send Email")
            
            if st.button("📤 Send Now", type="primary", use_container_width=True):
                return self._send_composed_email()
            
            st.markdown("---")
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            
            if st.button("💾 Save Draft", use_container_width=True):
                return self._save_draft()
            
            if st.button("👁️ Preview", use_container_width=True):
                return self._preview_email()
            
            if st.button("🧪 Test Send", use_container_width=True):
                return self._test_send()
        
        return {'status': 'composing', 'message': 'Email composition in progress'}
    
    def _compose_tab(self) -> Dict:
        """Compose email tab content."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Email subject
            self.composition_state['subject'] = st.text_input(
                "📌 Subject Line",
                value=self.composition_state['subject'],
                placeholder="Enter email subject...",
                help="Clear, actionable subject line for better open rates"
            )
            
            # Email body
            self.composition_state['body'] = st.text_area(
                "✍️ Email Content",
                value=self.composition_state['body'],
                height=300,
                placeholder="Compose your message here...\n\nTip: Use {{employee_name}}, {{risk_level}}, {{probability}} for personalization",
                help="HTML formatting supported. Use template variables for personalization."
            )
        
        with col2:
            # Email priority
            self.composition_state['priority'] = st.selectbox(
                "⚡ Priority",
                options=['low', 'normal', 'high', 'urgent'],
                index=1,
                help="Email priority level"
            )
            
            # Scheduled sending
            self.composition_state['send_later'] = st.checkbox(
                "⏰ Schedule Send",
                help="Schedule email for later delivery"
            )
            
            if self.composition_state['send_later']:
                self.composition_state['scheduled_time'] = st.datetime_input(
                    "Send Time",
                    value=datetime.now() + timedelta(hours=1),
                    min_value=datetime.now(),
                    help="When to send the email"
                )
            
            # Attachments
            st.markdown("### 📎 Attachments")
            uploaded_files = st.file_uploader(
                "Upload files",
                accept_multiple_files=True,
                type=['pdf', 'csv', 'xlsx', 'png', 'jpg'],
                help="Max 10MB per file"
            )
            
            if uploaded_files:
                self.composition_state['attachments'] = uploaded_files
                st.success(f"📎 {len(uploaded_files)} file(s) attached")
        
        # Email preview
        if self.composition_state['subject'] or self.composition_state['body']:
            st.markdown("### 👁️ Email Preview")
            with st.expander("Preview Email", expanded=False):
                self._render_email_preview()
        
        return {'status': 'draft_updated'}
    
    def _templates_tab(self):
        """Email templates selection tab."""
        st.markdown("### 📋 Email Templates")
        
        # Template categories
        template_categories = {
            '🚨 High Risk Alerts': [
                'high_risk_employee_alert',
                'immediate_intervention_required',
                'manager_urgent_notification'
            ],
            '⚠️ Medium Risk Notifications': [
                'medium_risk_employee_notice',
                'check_in_reminder',
                'development_opportunity'
            ],
            '📊 Periodic Reports': [
                'weekly_attrition_report',
                'monthly_analytics_summary',
                'quarterly_review_report'
            ],
            '🎯 Action Items': [
                'intervention_follow_up',
                'meeting_request',
                'survey_invitation'
            ]
        }
        
        # Template selector
        selected_category = st.selectbox(
            "Select Category",
            options=list(template_categories.keys()),
            help="Choose template category"
        )
        
        if selected_category:
            templates = template_categories[selected_category]
            selected_template = st.selectbox(
                "Select Template",
                options=templates,
                help="Choose specific template"
            )
            
            if selected_template:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Load and display template
                    template_data = self._load_template(selected_template)
                    if template_data:
                        st.markdown(f"**Template: {template_data['name']}**")
                        st.markdown(f"*{template_data['description']}*")
                        
                        # Template preview
                        with st.expander("Template Preview", expanded=True):
                            st.markdown(f"**Subject:** {template_data['subject']}")
                            st.markdown("**Body:**")
                            st.markdown(template_data['body'][:500] + "..." if len(template_data['body']) > 500 else template_data['body'])
                
                with col2:
                    if st.button("📥 Use Template", type="primary"):
                        self._apply_template(template_data)
                        st.success("✅ Template applied!")
                        st.rerun()
                    
                    if st.button("✏️ Customize"):
                        self._customize_template(template_data)
                    
                    st.markdown(f"**Variables:**")
                    if 'variables' in template_data:
                        for var in template_data['variables']:
                            st.markdown(f"• `{{{{{var}}}}}`")
    
    def _recipients_tab(self):
        """Recipients management tab."""
        st.markdown("### 👥 Email Recipients")
        
        # Recipient input methods
        input_method = st.radio(
            "Select Input Method",
            options=["✍️ Manual Entry", "📊 From Predictions", "📁 Upload List", "🎯 Smart Selection"],
            horizontal=True
        )
        
        if input_method == "✍️ Manual Entry":
            self._manual_recipients_input()
        
        elif input_method == "📊 From Predictions":
            self._prediction_based_recipients()
        
        elif input_method == "📁 Upload List":
            self._upload_recipients_list()
        
        elif input_method == "🎯 Smart Selection":
            self._smart_recipient_selection()
        
        # Current recipients display
        if self.composition_state['recipients']:
            st.markdown("### 📋 Current Recipients")
            recipients_df = pd.DataFrame(self.composition_state['recipients'])
            st.dataframe(recipients_df, use_container_width=True)
            
            st.markdown(f"**Total Recipients:** {len(self.composition_state['recipients'])}")
    
    def _settings_tab(self):
        """Email settings tab."""
        st.markdown("### ⚙️ Email Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📤 Sending Options")
            
            # Sender information
            sender_name = st.text_input(
                "Sender Name",
                value="HR Analytics Team",
                help="Display name for sender"
            )
            
            sender_email = st.text_input(
                "Reply-to Email",
                value="hr-analytics@company.com",
                help="Reply-to email address"
            )
            
            # Email format
            email_format = st.selectbox(
                "Email Format",
                options=['HTML', 'Plain Text', 'Mixed'],
                help="Email content format"
            )
            
            # Tracking options
            st.markdown("#### 📊 Tracking")
            
            track_opens = st.checkbox(
                "Track Email Opens",
                value=True,
                help="Track when emails are opened"
            )
            
            track_clicks = st.checkbox(
                "Track Link Clicks",
                value=True,
                help="Track clicks on links in emails"
            )
        
        with col2:
            st.markdown("#### 🚀 Delivery Options")
            
            # Batch sending
            enable_batching = st.checkbox(
                "Enable Batch Sending",
                value=True,
                help="Send emails in batches to avoid rate limits"
            )
            
            if enable_batching:
                batch_size = st.slider(
                    "Batch Size",
                    min_value=10,
                    max_value=100,
                    value=50,
                    help="Number of emails per batch"
                )
                
                batch_interval = st.slider(
                    "Interval Between Batches (seconds)",
                    min_value=1,
                    max_value=60,
                    value=5,
                    help="Wait time between batches"
                )
            
            # Retry settings
            st.markdown("#### 🔄 Retry Settings")
            
            max_retries = st.slider(
                "Max Retries",
                min_value=0,
                max_value=5,
                value=3,
                help="Maximum retry attempts for failed sends"
            )
            
            retry_delay = st.slider(
                "Retry Delay (minutes)",
                min_value=1,
                max_value=30,
                value=5,
                help="Wait time before retrying failed sends"
            )
    
    def template_selector(self, category: str = None, context: Dict = None) -> Dict:
        """
        Email template selection interface.
        
        Args:
            category: Template category filter
            context: Context data for template personalization
            
        Returns:
            dict: Selected template data
        """
        st.markdown("### 📋 Template Selector")
        
        # Available templates
        templates = self._get_available_templates(category)
        
        if not templates:
            st.warning("No templates available for the selected category")
            return {}
        
        # Template selection grid
        cols = st.columns(min(3, len(templates)))
        selected_template = None
        
        for idx, template in enumerate(templates):
            with cols[idx % 3]:
                with st.container():
                    st.markdown(f"""
                    <div style='border: 1px solid #ddd; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
                        <h4>{template['name']}</h4>
                        <p style='color: #666; font-size: 0.9em;'>{template['description']}</p>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='color: #888; font-size: 0.8em;'>{template['category']}</span>
                            <span style='color: #007bff; font-size: 0.8em;'>★ {template.get('rating', 4.5)}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Select {template['name']}", key=f"select_{idx}"):
                        selected_template = template
        
        # Template preview and customization
        if selected_template:
            st.markdown("### 👁️ Template Preview")
            
            # Apply context if provided
            if context:
                preview_template = self._apply_template_context(selected_template, context)
            else:
                preview_template = selected_template
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Subject preview
                st.text_input(
                    "Subject",
                    value=preview_template['subject'],
                    disabled=True
                )
                
                # Body preview
                st.text_area(
                    "Body",
                    value=preview_template['body'],
                    height=200,
                    disabled=True
                )
            
            with col2:
                st.markdown("**Template Info**")
                st.markdown(f"Category: {preview_template['category']}")
                st.markdown(f"Language: {preview_template.get('language', 'English')}")
                st.markdown(f"Type: {preview_template.get('type', 'HTML')}")
                
                if 'variables' in preview_template:
                    st.markdown("**Variables:**")
                    for var in preview_template['variables']:
                        st.markdown(f"• `{{{{{var}}}}}`")
        
        return selected_template or {}
    
    def bulk_email_sender(self, 
                         recipients: Union[List[Dict], pd.DataFrame],
                         template: Dict,
                         personalization_data: Dict = None) -> Dict:
        """
        Bulk email sending interface and operations.
        
        Args:
            recipients: List of recipient information
            template: Email template to use
            personalization_data: Data for personalizing emails
            
        Returns:
            dict: Bulk send results and status
        """
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                    padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h2 style='color: white; margin: 0; text-align: center;'>
                📬 Bulk Email Sender
            </h2>
            <p style='color: white; margin: 0.5rem 0 0 0; text-align: center; opacity: 0.9;'>
                Send personalized emails to multiple recipients
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Convert recipients to DataFrame if needed
        if isinstance(recipients, list):
            recipients_df = pd.DataFrame(recipients)
        else:
            recipients_df = recipients.copy()
        
        # Bulk sending configuration
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Recipients Overview")
            
            # Recipients summary
            st.dataframe(recipients_df.head(), use_container_width=True)
            st.markdown(f"**Total Recipients:** {len(recipients_df)}")
            
            # Template preview
            st.markdown("### 📋 Template Preview")
            st.markdown(f"**Subject:** {template.get('subject', 'No subject')}")
            
            with st.expander("Email Body Preview", expanded=False):
                st.markdown(template.get('body', 'No body content'))
        
        with col2:
            st.markdown("### ⚙️ Bulk Send Settings")
            
            # Batch configuration
            batch_size = st.slider(
                "Batch Size",
                min_value=10,
                max_value=200,
                value=self.bulk_state['batch_size'],
                help="Emails per batch"
            )
            
            send_interval = st.slider(
                "Batch Interval (seconds)",
                min_value=1,
                max_value=60,
                value=self.bulk_state['send_interval'],
                help="Wait between batches"
            )
            
            # Personalization settings
            st.markdown("#### 🎯 Personalization")
            
            personalize_emails = st.checkbox(
                "Enable Personalization",
                value=True,
                help="Personalize emails with recipient data"
            )
            
            if personalize_emails and personalization_data:
                st.markdown("**Available Fields:**")
                for field in personalization_data.keys():
                    st.markdown(f"• `{{{{{field}}}}}`")
            
            # Test sending
            st.markdown("#### 🧪 Testing")
            
            test_recipient = st.text_input(
                "Test Email",
                placeholder="test@company.com",
                help="Send test email before bulk send"
            )
            
            if st.button("📤 Send Test", type="secondary"):
                if test_recipient:
                    test_result = self._send_test_bulk_email(
                        test_recipient, template, personalization_data
                    )
                    if test_result['success']:
                        st.success("✅ Test email sent successfully!")
                    else:
                        st.error(f"❌ Test failed: {test_result['error']}")
        
        # Bulk send execution
        st.markdown("### 🚀 Execute Bulk Send")
        
        # Confirmation and warnings
        if len(recipients_df) > 100:
            st.warning(f"⚠️ You are about to send {len(recipients_df)} emails. This may take some time.")
        
        # Cost estimation (if available)
        if hasattr(self, '_estimate_sending_cost'):
            estimated_cost = self._estimate_sending_cost(len(recipients_df))
            st.info(f"💰 Estimated cost: ${estimated_cost:.2f}")
        
        # Progress tracking
        if 'bulk_send_progress' in st.session_state:
            progress = st.session_state.bulk_send_progress
            st.progress(progress['completed'] / progress['total'])
            st.markdown(f"Progress: {progress['completed']}/{progress['total']} emails sent")
        
        # Send button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🚀 Start Bulk Send", type="primary", use_container_width=True):
                return self._execute_bulk_send(
                    recipients_df, template, personalization_data,
                    batch_size, send_interval
                )
        
        with col2:
            if st.button("⏸️ Pause Sending", use_container_width=True):
                return self._pause_bulk_send()
        
        with col3:
            if st.button("🛑 Cancel Sending", use_container_width=True):
                return self._cancel_bulk_send()
        
        return {'status': 'ready', 'recipients_count': len(recipients_df)}
    
    # Helper Methods
    def _load_template(self, template_name: str) -> Dict:
        """Load email template by name."""
        if self.templates:
            return self.templates.get_template(template_name)
        
        # Fallback templates
        templates = {
            'high_risk_employee_alert': {
                'name': 'High Risk Employee Alert',
                'description': 'Alert for employees with high attrition risk',
                'subject': '🚨 High Attrition Risk Alert - {{employee_name}}',
                'body': '''
                <h2>High Attrition Risk Alert</h2>
                <p>Dear {{manager_name}},</p>
                <p>Our predictive analytics has identified <strong>{{employee_name}}</strong> as having a high risk of attrition.</p>
                <ul>
                    <li><strong>Risk Level:</strong> {{risk_level}}</li>
                    <li><strong>Probability:</strong> {{probability}}%</li>
                    <li><strong>Department:</strong> {{department}}</li>
                </ul>
                <p><strong>Immediate Action Required:</strong></p>
                <ul>
                    <li>Schedule a one-on-one meeting within 48 hours</li>
                    <li>Review recent feedback and performance</li>
                    <li>Consider retention incentives</li>
                </ul>
                <p>Best regards,<br>HR Analytics Team</p>
                ''',
                'category': '🚨 High Risk Alerts',
                'variables': ['employee_name', 'manager_name', 'risk_level', 'probability', 'department']
            }
        }
        
        return templates.get(template_name, {})
    
    def _apply_template(self, template_data: Dict):
        """Apply template to composition state."""
        self.composition_state['subject'] = template_data.get('subject', '')
        self.composition_state['body'] = template_data.get('body', '')
        self.composition_state['template'] = template_data
    
    def _render_email_preview(self):
        """Render email preview."""
        st.markdown(f"**Subject:** {self.composition_state['subject']}")
        st.markdown("**Body:**")
        
        # Render HTML content safely
        body = self.composition_state['body']
        if '<html>' in body.lower() or '<div>' in body.lower():
            st.markdown(body, unsafe_allow_html=True)
        else:
            st.markdown(body)
    
    def _send_composed_email(self) -> Dict:
        """Send the composed email."""
        try:
            if not self.email_service:
                return {'success': False, 'error': 'Email service not available'}
            
            # Validate composition
            if not self.composition_state['recipients']:
                return {'success': False, 'error': 'No recipients specified'}
            
            if not self.composition_state['subject']:
                return {'success': False, 'error': 'Subject line required'}
            
            # Send email
            result = self.email_service.send_email(
                recipients=self.composition_state['recipients'],
                subject=self.composition_state['subject'],
                body=self.composition_state['body'],
                attachments=self.composition_state.get('attachments', []),
                priority=self.composition_state.get('priority', 'normal'),
                scheduled_time=self.composition_state.get('scheduled_time')
            )
            
            if result['success']:
                st.success("✅ Email sent successfully!")
                # Clear composition state
                self.composition_state = {
                    'recipients': [],
                    'subject': '',
                    'body': '',
                    'template': None,
                    'attachments': [],
                    'priority': 'normal',
                    'send_later': False,
                    'scheduled_time': None
                }
            else:
                st.error(f"❌ Failed to send email: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_bulk_send(self, recipients_df: pd.DataFrame, template: Dict,
                          personalization_data: Dict, batch_size: int, send_interval: int) -> Dict:
        """Execute bulk email sending."""
        try:
            if not self.email_service:
                return {'success': False, 'error': 'Email service not available'}
            
            # Initialize progress tracking
            st.session_state.bulk_send_progress = {
                'total': len(recipients_df),
                'completed': 0,
                'failed': 0,
                'status': 'running'
            }
            
            # Process in batches
            results = {'sent': 0, 'failed': 0, 'errors': []}
            
            for batch_start in range(0, len(recipients_df), batch_size):
                batch_end = min(batch_start + batch_size, len(recipients_df))
                batch_recipients = recipients_df.iloc[batch_start:batch_end]
                
                # Send batch
                batch_results = self.email_service.send_bulk_email(
                    recipients=batch_recipients.to_dict('records'),
                    template=template,
                    personalization_data=personalization_data
                )
                
                # Update results
                results['sent'] += batch_results.get('sent', 0)
                results['failed'] += batch_results.get('failed', 0)
                results['errors'].extend(batch_results.get('errors', []))
                
                # Update progress
                st.session_state.bulk_send_progress['completed'] = batch_end
                
                # Wait between batches
                if batch_end < len(recipients_df):
                    time.sleep(send_interval)
            
            # Final status
            st.session_state.bulk_send_progress['status'] = 'completed'
            
            success_message = f"✅ Bulk send completed! Sent: {results['sent']}, Failed: {results['failed']}"
            st.success(success_message)
            
            return {
                'success': True,
                'sent': results['sent'],
                'failed': results['failed'],
                'errors': results['errors']
            }
            
        except Exception as e:
            logger.error(f"Error in bulk send: {e}")
            st.error(f"❌ Bulk send failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _manual_recipients_input(self):
        """Manual recipient input interface."""
        st.markdown("#### ✍️ Add Recipients Manually")
        
        # Single recipient input
        with st.form("add_recipient"):
            col1, col2 = st.columns(2)
            
            with col1:
                email = st.text_input("Email Address", placeholder="employee@company.com")
                name = st.text_input("Full Name", placeholder="John Doe")
                department = st.selectbox("Department", 
                    options=['Engineering', 'Sales', 'HR', 'Marketing', 'Finance', 'Operations'])
            
            with col2:
                role = st.text_input("Job Role", placeholder="Senior Developer")
                manager = st.text_input("Manager", placeholder="Jane Smith")
                employee_id = st.text_input("Employee ID", placeholder="EMP001")
            
            if st.form_submit_button("➕ Add Recipient"):
                if email and name:
                    recipient = {
                        'email': email,
                        'name': name,
                        'department': department,
                        'role': role,
                        'manager': manager,
                        'employee_id': employee_id
                    }
                    self.composition_state['recipients'].append(recipient)
                    st.success(f"✅ Added {name} to recipients")
                else:
                    st.error("Email and name are required")
    
    def _prediction_based_recipients(self):
        """Load recipients from prediction results."""
        st.markdown("#### 📊 Load from Prediction Results")
        
        # File uploader for prediction results
        uploaded_file = st.file_uploader(
            "Upload Prediction Results CSV",
            type=['csv'],
            help="Upload CSV file with prediction results"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head(), use_container_width=True)
                
                # Risk level filter
                if 'risk_level' in df.columns or 'Risk_Level' in df.columns:
                    risk_col = 'risk_level' if 'risk_level' in df.columns else 'Risk_Level'
                    
                    selected_risks = st.multiselect(
                        "Select Risk Levels",
                        options=df[risk_col].unique(),
                        default=df[risk_col].unique(),
                        help="Filter recipients by risk level"
                    )
                    
                    if selected_risks:
                        filtered_df = df[df[risk_col].isin(selected_risks)]
                        
                        if st.button(f"📥 Load {len(filtered_df)} Recipients"):
                            recipients = []
                            for _, row in filtered_df.iterrows():
                                recipient = {
                                    'email': row.get('email', ''),
                                    'name': row.get('name', row.get('employee_name', '')),
                                    'department': row.get('department', ''),
                                    'risk_level': row.get(risk_col, ''),
                                    'probability': row.get('probability', 0),
                                    'employee_id': row.get('employee_id', '')
                                }
                                recipients.append(recipient)
                            
                            self.composition_state['recipients'].extend(recipients)
                            st.success(f"✅ Loaded {len(recipients)} recipients from predictions")
                
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    def _get_available_templates(self, category: str = None) -> List[Dict]:
        """Get available email templates."""
        if self.templates:
            return self.templates.get_templates_by_category(category)
        
        # Fallback templates
        default_templates = [
            {
                'name': 'High Risk Alert',
                'description': 'Alert for high-risk employees',
                'category': '🚨 High Risk Alerts',
                'subject': '🚨 High Attrition Risk - {{employee_name}}',
                'body': 'High risk alert template body...',
                'variables': ['employee_name', 'risk_level', 'probability'],
                'rating': 4.8
            },
            {
                'name': 'Weekly Report',
                'description': 'Weekly attrition analytics report',
                'category': '📊 Periodic Reports',
                'subject': '📊 Weekly Attrition Report - {{date}}',
                'body': 'Weekly report template body...',
                'variables': ['date', 'total_employees', 'high_risk_count'],
                'rating': 4.5
            }
        ]
        
        if category:
            return [t for t in default_templates if t['category'] == category]
        return default_templates

# Streamlit App Interface (if run directly)
def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="HR Email Sender",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📧 HR Email Sender")
    st.markdown("Frontend email interface for HR notifications and reports")
    
    # Initialize email sender
    email_sender = EmailSender()
    
    # Main interface
    tab1, tab2 = st.tabs(["📝 Compose Email", "📬 Bulk Sender"])
    
    with tab1:
        email_sender.compose_email_interface()
    
    with tab2:
        # Mock data for bulk sender demo
        if st.button("Load Sample Recipients"):
            sample_recipients = [
                {'email': 'john@company.com', 'name': 'John Doe', 'risk_level': 'High'},
                {'email': 'jane@company.com', 'name': 'Jane Smith', 'risk_level': 'Medium'}
            ]
            sample_template = email_sender._load_template('high_risk_employee_alert')
            email_sender.bulk_email_sender(sample_recipients, sample_template)

if __name__ == "__main__":
    main()
