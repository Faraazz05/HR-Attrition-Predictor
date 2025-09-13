"""
HR Attrition Predictor - Email Automation Service
===============================================
Comprehensive email automation for HR notifications, manager alerts,
engagement surveys, and retention campaigns with template management.

Author: HR Analytics Team
Date: September 2025
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd
from jinja2 import Template, Environment, FileSystemLoader
import base64
from dataclasses import dataclass, asdict
import asyncio
import aiosmtplib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================================================================
# DATA CLASSES
# ================================================================

@dataclass
class SMTPConfig:
    """SMTP server configuration."""
    host: str
    port: int = 587
    username: str = ""
    password: str = ""
    use_tls: bool = True
    timeout: int = 30

@dataclass
class EmailRecipient:
    """Email recipient information."""
    email: str
    name: str = ""
    role: str = ""
    department: str = ""

@dataclass
class EmailTemplate:
    """Email template structure."""
    template_id: str
    subject: str
    html_content: str
    text_content: str = ""
    template_vars: Dict[str, Any] = None
    category: str = "general"

@dataclass
class EmailCampaign:
    """Email campaign configuration."""
    campaign_id: str
    name: str
    template_id: str
    recipients: List[EmailRecipient]
    schedule_time: Optional[datetime] = None
    priority: str = "normal"  # low, normal, high
    tracking_enabled: bool = True

@dataclass
class EmailResult:
    """Email sending result."""
    success: bool
    message: str
    recipient_email: str
    timestamp: datetime
    error_details: Optional[str] = None

# ================================================================
# EMAIL SERVICE CLASS
# ================================================================

class EmailService:
    """
    Comprehensive email automation service for HR communications.
    
    Features:
    - SMTP configuration and connection management
    - Template management with Jinja2
    - Automated HR alerts and notifications
    - Employee engagement surveys
    - Retention campaign automation
    - Bulk email sending with rate limiting
    - Email tracking and analytics
    """
    
    def __init__(self, smtp_config: SMTPConfig, templates_dir: str = "email_templates"):
        """
        Initialize EmailService with SMTP configuration.
        
        Args:
            smtp_config: SMTP server configuration
            templates_dir: Directory containing email templates
        """
        self.smtp_config = smtp_config
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        
        # Template environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True
        )
        
        # Email templates storage
        self.templates: Dict[str, EmailTemplate] = {}
        
        # Email sending statistics
        self.email_stats = {
            'sent': 0,
            'failed': 0,
            'total_campaigns': 0,
            'last_sent': None
        }
        
        # Rate limiting (emails per minute)
        self.rate_limit = 60
        self.sent_times = []
        
        logger.info(f"EmailService initialized with SMTP: {smtp_config.host}:{smtp_config.port}")
        
        # Create default templates
        self.create_email_templates()
    
    def _check_smtp_connection(self) -> bool:
        """Test SMTP connection."""
        try:
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_config.host, self.smtp_config.port, timeout=self.smtp_config.timeout) as server:
                if self.smtp_config.use_tls:
                    server.starttls(context=context)
                
                if self.smtp_config.username and self.smtp_config.password:
                    server.login(self.smtp_config.username, self.smtp_config.password)
                
                logger.info("✅ SMTP connection successful")
                return True
        
        except Exception as e:
            logger.error(f"❌ SMTP connection failed: {e}")
            return False
    
    def _rate_limit_check(self) -> bool:
        """Check if rate limit allows sending email."""
        now = datetime.now()
        
        # Remove timestamps older than 1 minute
        self.sent_times = [t for t in self.sent_times if (now - t).seconds < 60]
        
        if len(self.sent_times) >= self.rate_limit:
            logger.warning(f"Rate limit reached: {len(self.sent_times)}/min")
            return False
        
        return True
    
    def _send_single_email(self, recipient: EmailRecipient, template: EmailTemplate, 
                          template_vars: Optional[Dict[str, Any]] = None) -> EmailResult:
        """Send a single email."""
        
        try:
            # Rate limiting
            if not self._rate_limit_check():
                return EmailResult(
                    success=False,
                    message="Rate limit exceeded",
                    recipient_email=recipient.email,
                    timestamp=datetime.now(),
                    error_details="Too many emails sent in the last minute"
                )
            
            # Prepare template variables
            vars_dict = template_vars or {}
            vars_dict.update({
                'recipient_name': recipient.name or recipient.email,
                'recipient_email': recipient.email,
                'current_date': datetime.now().strftime('%Y-%m-%d'),
                'current_year': datetime.now().year
            })
            
            # Render template
            subject = Template(template.subject).render(**vars_dict)
            
            if template.html_content:
                html_body = Template(template.html_content).render(**vars_dict)
            else:
                html_body = None
            
            if template.text_content:
                text_body = Template(template.text_content).render(**vars_dict)
            else:
                text_body = None
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.smtp_config.username
            msg['To'] = recipient.email
            msg['Subject'] = subject
            
            # Add text version
            if text_body:
                msg.attach(MIMEText(text_body, 'plain', 'utf-8'))
            
            # Add HTML version
            if html_body:
                msg.attach(MIMEText(html_body, 'html', 'utf-8'))
            
            # Send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_config.host, self.smtp_config.port, timeout=self.smtp_config.timeout) as server:
                if self.smtp_config.use_tls:
                    server.starttls(context=context)
                
                if self.smtp_config.username and self.smtp_config.password:
                    server.login(self.smtp_config.username, self.smtp_config.password)
                
                server.send_message(msg)
            
            # Track sending time
            self.sent_times.append(datetime.now())
            self.email_stats['sent'] += 1
            self.email_stats['last_sent'] = datetime.now()
            
            logger.info(f"✅ Email sent successfully to {recipient.email}")
            
            return EmailResult(
                success=True,
                message="Email sent successfully",
                recipient_email=recipient.email,
                timestamp=datetime.now()
            )
        
        except Exception as e:
            self.email_stats['failed'] += 1
            logger.error(f"❌ Failed to send email to {recipient.email}: {e}")
            
            return EmailResult(
                success=False,
                message=f"Failed to send email: {str(e)}",
                recipient_email=recipient.email,
                timestamp=datetime.now(),
                error_details=str(e)
            )
    
    def send_manager_alert(self, manager_email: str, manager_name: str,
                          high_risk_employees: List[Dict[str, Any]],
                          department: str = "") -> EmailResult:
        """
        Send manager alert about high-risk employees.
        
        Args:
            manager_email: Manager's email address
            manager_name: Manager's name
            high_risk_employees: List of high-risk employee data
            department: Department name
            
        Returns:
            EmailResult with sending status
        """
        
        logger.info(f"📧 Sending manager alert to {manager_email}")
        
        # Prepare recipient
        recipient = EmailRecipient(
            email=manager_email,
            name=manager_name,
            role="Manager",
            department=department
        )
        
        # Get template
        template = self.templates.get('manager_alert')
        if not template:
            return EmailResult(
                success=False,
                message="Manager alert template not found",
                recipient_email=manager_email,
                timestamp=datetime.now(),
                error_details="Template 'manager_alert' is missing"
            )
        
        # Template variables
        template_vars = {
            'manager_name': manager_name,
            'department': department,
            'high_risk_employees': high_risk_employees,
            'risk_count': len(high_risk_employees),
            'alert_date': datetime.now().strftime('%B %d, %Y'),
            'dashboard_url': os.getenv('DASHBOARD_URL', 'https://your-dashboard.com'),
            'support_email': os.getenv('HR_SUPPORT_EMAIL', 'hr-support@company.com')
        }
        
        return self._send_single_email(recipient, template, template_vars)
    
    def send_employee_engagement_survey(self, employees: List[Dict[str, Any]]) -> List[EmailResult]:
        """
        Send employee engagement surveys to multiple employees.
        
        Args:
            employees: List of employee dictionaries with email, name, etc.
            
        Returns:
            List of EmailResult objects
        """
        
        logger.info(f"📧 Sending engagement surveys to {len(employees)} employees")
        
        # Get template
        template = self.templates.get('engagement_survey')
        if not template:
            return [EmailResult(
                success=False,
                message="Engagement survey template not found",
                recipient_email=emp.get('email', 'unknown'),
                timestamp=datetime.now(),
                error_details="Template 'engagement_survey' is missing"
            ) for emp in employees]
        
        results = []
        
        for employee in employees:
            # Prepare recipient
            recipient = EmailRecipient(
                email=employee.get('email', ''),
                name=employee.get('name', ''),
                role=employee.get('role', ''),
                department=employee.get('department', '')
            )
            
            # Skip if no email
            if not recipient.email:
                results.append(EmailResult(
                    success=False,
                    message="No email address provided",
                    recipient_email="",
                    timestamp=datetime.now(),
                    error_details="Employee email is missing"
                ))
                continue
            
            # Template variables
            template_vars = {
                'employee_name': recipient.name,
                'employee_role': recipient.role,
                'department': recipient.department,
                'survey_url': f"{os.getenv('SURVEY_BASE_URL', 'https://survey.company.com')}/engagement/{employee.get('employee_id', 'default')}",
                'survey_deadline': (datetime.now() + timedelta(days=7)).strftime('%B %d, %Y'),
                'hr_contact': os.getenv('HR_CONTACT_EMAIL', 'hr@company.com')
            }
            
            result = self._send_single_email(recipient, template, template_vars)
            results.append(result)
            
            # Small delay to avoid overwhelming SMTP server
            import time
            time.sleep(0.1)
        
        successful_sends = sum(1 for r in results if r.success)
        logger.info(f"📧 Engagement survey campaign completed: {successful_sends}/{len(employees)} successful")
        
        return results
    
    def send_retention_campaign(self, at_risk_employees: List[Dict[str, Any]], 
                               campaign_type: str = "general") -> List[EmailResult]:
        """
        Send personalized retention campaign emails to at-risk employees.
        
        Args:
            at_risk_employees: List of at-risk employee data
            campaign_type: Type of retention campaign (general, benefits, career, wellness)
            
        Returns:
            List of EmailResult objects
        """
        
        logger.info(f"📧 Sending {campaign_type} retention campaign to {len(at_risk_employees)} employees")
        
        # Select appropriate template based on campaign type
        template_map = {
            'general': 'retention_general',
            'benefits': 'retention_benefits',
            'career': 'retention_career',
            'wellness': 'retention_wellness'
        }
        
        template_id = template_map.get(campaign_type, 'retention_general')
        template = self.templates.get(template_id)
        
        if not template:
            return [EmailResult(
                success=False,
                message=f"Retention template '{template_id}' not found",
                recipient_email=emp.get('email', 'unknown'),
                timestamp=datetime.now(),
                error_details=f"Template '{template_id}' is missing"
            ) for emp in at_risk_employees]
        
        results = []
        
        for employee in at_risk_employees:
            # Prepare recipient
            recipient = EmailRecipient(
                email=employee.get('email', ''),
                name=employee.get('name', ''),
                role=employee.get('role', ''),
                department=employee.get('department', '')
            )
            
            if not recipient.email:
                results.append(EmailResult(
                    success=False,
                    message="No email address provided",
                    recipient_email="",
                    timestamp=datetime.now(),
                    error_details="Employee email is missing"
                ))
                continue
            
            # Personalized template variables based on employee data
            template_vars = {
                'employee_name': recipient.name,
                'employee_role': recipient.role,
                'department': recipient.department,
                'tenure_years': employee.get('years_at_company', 1),
                'risk_level': employee.get('risk_level', 'Medium'),
                'manager_name': employee.get('manager_name', 'Your Manager'),
                'hr_contact': employee.get('hr_contact', 'hr@company.com'),
                'benefits_url': os.getenv('BENEFITS_PORTAL_URL', 'https://benefits.company.com'),
                'career_portal_url': os.getenv('CAREER_PORTAL_URL', 'https://careers.company.com'),
                'wellness_url': os.getenv('WELLNESS_PORTAL_URL', 'https://wellness.company.com'),
                'feedback_url': f"{os.getenv('FEEDBACK_URL', 'https://feedback.company.com')}/{employee.get('employee_id', 'default')}"
            }
            
            # Add campaign-specific variables
            if campaign_type == 'benefits':
                template_vars.update({
                    'new_benefits': ['Enhanced Health Insurance', 'Flexible PTO', 'Remote Work Options'],
                    'savings_estimate': '$2,500 annually'
                })
            elif campaign_type == 'career':
                template_vars.update({
                    'available_roles': employee.get('suggested_roles', ['Senior ' + recipient.role]),
                    'skill_development': employee.get('suggested_skills', ['Leadership', 'Technical Skills'])
                })
            elif campaign_type == 'wellness':
                template_vars.update({
                    'wellness_programs': ['Mental Health Support', 'Fitness Membership', 'Stress Management'],
                    'work_life_score': employee.get('work_life_balance', 3)
                })
            
            result = self._send_single_email(recipient, template, template_vars)
            results.append(result)
            
            # Delay between emails
            import time
            time.sleep(0.2)
        
        successful_sends = sum(1 for r in results if r.success)
        logger.info(f"📧 {campaign_type.title()} retention campaign completed: {successful_sends}/{len(at_risk_employees)} successful")
        
        return results
    
    def send_bulk_campaign(self, campaign: EmailCampaign, 
                          template_vars: Optional[Dict[str, Any]] = None) -> List[EmailResult]:
        """
        Send bulk email campaign to multiple recipients.
        
        Args:
            campaign: EmailCampaign configuration
            template_vars: Additional template variables
            
        Returns:
            List of EmailResult objects
        """
        
        logger.info(f"📧 Starting bulk campaign: {campaign.name}")
        
        template = self.templates.get(campaign.template_id)
        if not template:
            return [EmailResult(
                success=False,
                message=f"Template '{campaign.template_id}' not found",
                recipient_email=recipient.email,
                timestamp=datetime.now(),
                error_details=f"Template '{campaign.template_id}' is missing"
            ) for recipient in campaign.recipients]
        
        results = []
        
        # Schedule check
        if campaign.schedule_time and campaign.schedule_time > datetime.now():
            logger.info(f"Campaign scheduled for {campaign.schedule_time}")
            # In a production environment, you'd use a task scheduler like Celery
            return results
        
        for recipient in campaign.recipients:
            # Merge template variables
            vars_dict = template_vars or {}
            vars_dict.update({
                'campaign_name': campaign.name,
                'campaign_id': campaign.campaign_id
            })
            
            result = self._send_single_email(recipient, template, vars_dict)
            results.append(result)
            
            # Priority-based delays
            delay = 0.1 if campaign.priority == 'high' else 0.2 if campaign.priority == 'normal' else 0.5
            import time
            time.sleep(delay)
        
        self.email_stats['total_campaigns'] += 1
        
        successful_sends = sum(1 for r in results if r.success)
        logger.info(f"📧 Bulk campaign '{campaign.name}' completed: {successful_sends}/{len(campaign.recipients)} successful")
        
        return results
    
    def create_email_templates(self):
        """Create and register default email templates."""
        
        logger.info("📝 Creating default email templates")
        
        # Manager Alert Template
        self.templates['manager_alert'] = EmailTemplate(
            template_id='manager_alert',
            subject='🚨 HR Alert: High-Risk Employees in {{ department or "Your Team" }}',
            html_content=self._get_manager_alert_html(),
            text_content=self._get_manager_alert_text(),
            category='alert'
        )
        
        # Employee Engagement Survey Template
        self.templates['engagement_survey'] = EmailTemplate(
            template_id='engagement_survey',
            subject='📋 Your Voice Matters: Employee Engagement Survey',
            html_content=self._get_engagement_survey_html(),
            text_content=self._get_engagement_survey_text(),
            category='survey'
        )
        
        # Retention Campaign Templates
        self.templates['retention_general'] = EmailTemplate(
            template_id='retention_general',
            subject='💼 We Value You: Let\'s Talk About Your Journey at {{ company_name or "Our Company" }}',
            html_content=self._get_retention_general_html(),
            text_content=self._get_retention_general_text(),
            category='retention'
        )
        
        self.templates['retention_benefits'] = EmailTemplate(
            template_id='retention_benefits',
            subject='🎁 Exciting New Benefits Just for You!',
            html_content=self._get_retention_benefits_html(),
            text_content=self._get_retention_benefits_text(),
            category='retention'
        )
        
        self.templates['retention_career'] = EmailTemplate(
            template_id='retention_career',
            subject='🚀 Your Career Growth Opportunities Await',
            html_content=self._get_retention_career_html(),
            text_content=self._get_retention_career_text(),
            category='retention'
        )
        
        self.templates['retention_wellness'] = EmailTemplate(
            template_id='retention_wellness',
            subject='🌱 Prioritizing Your Well-being and Work-Life Balance',
            html_content=self._get_retention_wellness_html(),
            text_content=self._get_retention_wellness_text(),
            category='retention'
        )
        
        logger.info(f"📝 Created {len(self.templates)} email templates")
    
    def _get_manager_alert_html(self) -> str:
        """HTML template for manager alerts."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>HR Alert</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }
                .content { background: #f8f9fa; padding: 30px; }
                .alert-box { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; border-radius: 5px; }
                .employee-list { background: white; border-radius: 8px; overflow: hidden; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .employee-item { padding: 15px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
                .employee-item:last-child { border-bottom: none; }
                .risk-badge { padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; }
                .risk-high { background: #dc3545; color: white; }
                .risk-medium { background: #ffc107; color: black; }
                .btn { display: inline-block; padding: 12px 24px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 10px 0; }
                .footer { background: #343a40; color: white; padding: 20px; text-align: center; border-radius: 0 0 10px 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚨 HR Alert: High-Risk Employees</h1>
                    <p>{{ alert_date }}</p>
                </div>
                
                <div class="content">
                    <h2>Hello {{ manager_name }},</h2>
                    
                    <div class="alert-box">
                        <strong>⚠️ Immediate Attention Required</strong><br>
                        We've identified <strong>{{ risk_count }} employee(s)</strong> in 
                        {% if department %}{{ department }}{% else %}your team{% endif %} 
                        who are at high risk of attrition.
                    </div>
                    
                    <h3>High-Risk Employees:</h3>
                    <div class="employee-list">
                        {% for employee in high_risk_employees %}
                        <div class="employee-item">
                            <div>
                                <strong>{{ employee.name or employee.employee_id }}</strong><br>
                                <small>{{ employee.role or 'Employee' }} | {{ employee.department or 'N/A' }}</small>
                            </div>
                            <div>
                                <span class="risk-badge risk-{{ employee.risk_level|lower }}">
                                    {{ employee.risk_level|upper }} RISK
                                </span>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                    
                    <h3>🎯 Recommended Actions:</h3>
                    <ul>
                        <li><strong>Schedule one-on-one meetings</strong> within the next 48 hours</li>
                        <li><strong>Discuss career development</strong> and address any concerns</li>
                        <li><strong>Review compensation</strong> and benefits alignment</li>
                        <li><strong>Assess workload</strong> and work-life balance</li>
                        <li><strong>Document conversations</strong> and follow-up plans</li>
                    </ul>
                    
                    <a href="{{ dashboard_url }}" class="btn">📊 View Full Dashboard</a>
                    
                    <p><strong>Need Support?</strong> Contact HR at <a href="mailto:{{ support_email }}">{{ support_email }}</a></p>
                </div>
                
                <div class="footer">
                    <p>This alert was generated by the HR Attrition Prediction System<br>
                    <small>Confidential - For Management Use Only</small></p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_manager_alert_text(self) -> str:
        """Text template for manager alerts."""
        return """
        HR ALERT: High-Risk Employees
        {{ alert_date }}
        
        Hello {{ manager_name }},
        
        ⚠️ IMMEDIATE ATTENTION REQUIRED
        
        We've identified {{ risk_count }} employee(s) in {% if department %}{{ department }}{% else %}your team{% endif %} who are at high risk of attrition.
        
        HIGH-RISK EMPLOYEES:
        {% for employee in high_risk_employees %}
        - {{ employee.name or employee.employee_id }} ({{ employee.role or 'Employee' }}) - {{ employee.risk_level|upper }} RISK
        {% endfor %}
        
        RECOMMENDED ACTIONS:
        1. Schedule one-on-one meetings within 48 hours
        2. Discuss career development and address concerns
        3. Review compensation and benefits alignment
        4. Assess workload and work-life balance
        5. Document conversations and follow-up plans
        
        View Full Dashboard: {{ dashboard_url }}
        
        Need Support? Contact HR at {{ support_email }}
        
        ---
        This alert was generated by the HR Attrition Prediction System
        Confidential - For Management Use Only
        """
    
    def _get_engagement_survey_html(self) -> str:
        """HTML template for engagement surveys."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Employee Engagement Survey</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }
                .content { background: #f8f9fa; padding: 30px; }
                .highlight-box { background: #e7f3ff; border-left: 4px solid #007bff; padding: 20px; margin: 20px 0; border-radius: 5px; }
                .btn { display: inline-block; padding: 15px 30px; background: #28a745; color: white; text-decoration: none; border-radius: 25px; margin: 20px 0; font-weight: bold; text-align: center; }
                .features { display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0; }
                .feature { flex: 1; min-width: 150px; text-align: center; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .footer { background: #343a40; color: white; padding: 20px; text-align: center; border-radius: 0 0 10px 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📋 Your Voice Matters</h1>
                    <h2>Employee Engagement Survey</h2>
                </div>
                
                <div class="content">
                    <h2>Dear {{ employee_name }},</h2>
                    
                    <p>Your feedback is crucial to making our workplace even better! We're conducting our quarterly engagement survey to understand your experience and identify areas for improvement.</p>
                    
                    <div class="highlight-box">
                        <h3>🎯 Why Your Participation Matters</h3>
                        <p>Your honest feedback helps us:</p>
                        <ul>
                            <li>Improve workplace culture and environment</li>
                            <li>Enhance benefits and compensation</li>
                            <li>Develop better career growth opportunities</li>
                            <li>Address any concerns or challenges</li>
                        </ul>
                    </div>
                    
                    <div class="features">
                        <div class="feature">
                            <h4>⏱️ Quick</h4>
                            <p>Only 5-7 minutes to complete</p>
                        </div>
                        <div class="feature">
                            <h4>🔒 Anonymous</h4>
                            <p>Your responses are completely confidential</p>
                        </div>
                        <div class="feature">
                            <h4>📊 Impactful</h4>
                            <p>Your feedback drives real change</p>
                        </div>
                    </div>
                    
                    <div style="text-align: center;">
                        <a href="{{ survey_url }}" class="btn">🚀 Take Survey Now</a>
                    </div>
                    
                    <p><strong>Survey Deadline:</strong> {{ survey_deadline }}</p>
                    
                    <p>Questions? Contact HR at <a href="mailto:{{ hr_contact }}">{{ hr_contact }}</a></p>
                </div>
                
                <div class="footer">
                    <p>Thank you for helping us create a better workplace!<br>
                    <small>Survey powered by HR Analytics Platform</small></p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_engagement_survey_text(self) -> str:
        """Text template for engagement surveys."""
        return """
        YOUR VOICE MATTERS - Employee Engagement Survey
        
        Dear {{ employee_name }},
        
        Your feedback is crucial to making our workplace even better! We're conducting our quarterly engagement survey to understand your experience and identify areas for improvement.
        
        WHY YOUR PARTICIPATION MATTERS:
        Your honest feedback helps us:
        - Improve workplace culture and environment
        - Enhance benefits and compensation
        - Develop better career growth opportunities
        - Address any concerns or challenges
        
        SURVEY DETAILS:
        ⏱️ Quick: Only 5-7 minutes to complete
        🔒 Anonymous: Your responses are completely confidential  
        📊 Impactful: Your feedback drives real change
        
        TAKE SURVEY: {{ survey_url }}
        
        Survey Deadline: {{ survey_deadline }}
        
        Questions? Contact HR at {{ hr_contact }}
        
        Thank you for helping us create a better workplace!
        
        ---
        Survey powered by HR Analytics Platform
        """
    
    def _get_retention_general_html(self) -> str:
        """HTML template for general retention campaigns."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>We Value You</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }
                .content { background: #f8f9fa; padding: 30px; }
                .personal-note { background: #fff8e1; border-left: 4px solid #ff9800; padding: 20px; margin: 20px 0; border-radius: 5px; }
                .appreciation-box { background: white; padding: 25px; margin: 20px 0; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center; }
                .btn { display: inline-block; padding: 15px 30px; background: #007bff; color: white; text-decoration: none; border-radius: 25px; margin: 10px; font-weight: bold; }
                .footer { background: #343a40; color: white; padding: 20px; text-align: center; border-radius: 0 0 10px 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>💼 We Value You</h1>
                    <p>A message about your journey with us</p>
                </div>
                
                <div class="content">
                    <h2>Dear {{ employee_name }},</h2>
                    
                    <div class="appreciation-box">
                        <h3>🌟 Thank You for {{ tenure_years }} Amazing Years!</h3>
                        <p>Your contributions as a {{ employee_role }} have made a significant impact on our {{ department }} team and the company as a whole.</p>
                    </div>
                    
                    <p>We wanted to take a moment to reach out and let you know how much we appreciate your dedication, expertise, and the unique perspective you bring to our team.</p>
                    
                    <div class="personal-note">
                        <h3>💬 Let's Talk</h3>
                        <p>Your manager, {{ manager_name }}, would love to schedule some time to discuss:</p>
                        <ul>
                            <li>Your career goals and aspirations</li>
                            <li>Any challenges you're facing</li>
                            <li>Opportunities for growth and development</li>
                            <li>How we can better support you</li>
                        </ul>
                    </div>
                    
                    <h3>🚀 What's Next?</h3>
                    <p>We're committed to making sure your experience here continues to be rewarding and fulfilling. We have some exciting initiatives coming up and would love your input.</p>
                    
                    <div style="text-align: center;">
                        <a href="{{ feedback_url }}" class="btn">💭 Share Your Thoughts</a>
                        <a href="mailto:{{ manager_name }}" class="btn">📅 Schedule a Chat</a>
                    </div>
                    
                    <p>If you have any immediate concerns or questions, please don't hesitate to reach out to HR at <a href="mailto:{{ hr_contact }}">{{ hr_contact }}</a>.</p>
                </div>
                
                <div class="footer">
                    <p>Thank you for being an essential part of our team!<br>
                    <small>With appreciation from the HR Team</small></p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_retention_general_text(self) -> str:
        """Text template for general retention campaigns."""
        return """
        WE VALUE YOU - A Message About Your Journey With Us
        
        Dear {{ employee_name }},
        
        🌟 THANK YOU FOR {{ tenure_years }} AMAZING YEARS!
        
        Your contributions as a {{ employee_role }} have made a significant impact on our {{ department }} team and the company as a whole.
        
        We wanted to take a moment to reach out and let you know how much we appreciate your dedication, expertise, and the unique perspective you bring to our team.
        
        LET'S TALK
        Your manager, {{ manager_name }}, would love to schedule some time to discuss:
        - Your career goals and aspirations
        - Any challenges you're facing
        - Opportunities for growth and development  
        - How we can better support you
        
        WHAT'S NEXT?
        We're committed to making sure your experience here continues to be rewarding and fulfilling. We have some exciting initiatives coming up and would love your input.
        
        Share Your Thoughts: {{ feedback_url }}
        Schedule a Chat: Contact {{ manager_name }}
        
        If you have any immediate concerns or questions, please reach out to HR at {{ hr_contact }}.
        
        Thank you for being an essential part of our team!
        
        ---
        With appreciation from the HR Team
        """
    
    def _get_retention_benefits_html(self) -> str:
        """HTML template for benefits-focused retention."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Exciting New Benefits</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }
                .content { background: #f8f9fa; padding: 30px; }
                .benefits-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
                .benefit-card { background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.1); }
                .benefit-icon { font-size: 40px; margin-bottom: 15px; }
                .savings-highlight { background: #e8f5e8; border: 2px solid #28a745; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0; }
                .btn { display: inline-block; padding: 15px 30px; background: #28a745; color: white; text-decoration: none; border-radius: 25px; margin: 10px; font-weight: bold; }
                .footer { background: #343a40; color: white; padding: 20px; text-align: center; border-radius: 0 0 10px 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎁 Exciting New Benefits</h1>
                    <p>Exclusively designed for valued employees like you</p>
                </div>
                
                <div class="content">
                    <h2>Dear {{ employee_name }},</h2>
                    
                    <p>We're thrilled to announce some amazing new benefits that are now available to you! These enhancements are part of our commitment to supporting your well-being and professional growth.</p>
                    
                    <div class="benefits-grid">
                        {% for benefit in new_benefits %}
                        <div class="benefit-card">
                            <div class="benefit-icon">✨</div>
                            <h3>{{ benefit }}</h3>
                            <p>Now available to you</p>
                        </div>
                        {% endfor %}
                    </div>
                    
                    <div class="savings-highlight">
                        <h3>💰 Your Estimated Annual Savings</h3>
                        <p style="font-size: 24px; font-weight: bold; color: #28a745;">{{ savings_estimate }}</p>
                        <p>Based on your current role and benefit elections</p>
                    </div>
                    
                    <h3>🚀 How to Get Started</h3>
                    <p>These benefits are already active for your account! Visit our benefits portal to:</p>
                    <ul>
                        <li>Explore all available options</li>
                        <li>Make your benefit selections</li>
                        <li>Download helpful resources</li>
                        <li>Schedule a benefits consultation</li>
                    </ul>
                    
                    <div style="text-align: center;">
                        <a href="{{ benefits_url }}" class="btn">🔗 Access Benefits Portal</a>
                    </div>
                    
                    <p>Questions about your benefits? Contact HR at <a href="mailto:{{ hr_contact }}">{{ hr_contact }}</a></p>
                </div>
                
                <div class="footer">
                    <p>Investing in your well-being and future<br>
                    <small>Your HR Benefits Team</small></p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_retention_benefits_text(self) -> str:
        """Text template for benefits-focused retention."""
        return """
        EXCITING NEW BENEFITS - Exclusively for You!
        
        Dear {{ employee_name }},
        
        We're thrilled to announce some amazing new benefits that are now available to you! These enhancements are part of our commitment to supporting your well-being and professional growth.
        
        NEW BENEFITS AVAILABLE:
        {% for benefit in new_benefits %}
        ✨ {{ benefit }}
        {% endfor %}
        
        💰 YOUR ESTIMATED ANNUAL SAVINGS: {{ savings_estimate }}
        Based on your current role and benefit elections
        
        HOW TO GET STARTED:
        These benefits are already active for your account! Visit our benefits portal to:
        - Explore all available options
        - Make your benefit selections  
        - Download helpful resources
        - Schedule a benefits consultation
        
        Access Benefits Portal: {{ benefits_url }}
        
        Questions about your benefits? Contact HR at {{ hr_contact }}
        
        ---
        Investing in your well-being and future
        Your HR Benefits Team
        """
    
    def _get_retention_career_html(self) -> str:
        """HTML template for career-focused retention."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Your Career Growth</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }
                .content { background: #f8f9fa; padding: 30px; }
                .career-path { background: white; padding: 25px; margin: 20px 0; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
                .opportunities { display: grid; gap: 15px; margin: 20px 0; }
                .opportunity { background: #f0f8ff; padding: 15px; border-left: 4px solid #007bff; border-radius: 5px; }
                .skills-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 15px 0; }
                .skill-tag { background: #e3f2fd; color: #1976d2; padding: 8px 16px; border-radius: 20px; text-align: center; font-size: 14px; }
                .btn { display: inline-block; padding: 15px 30px; background: #6f42c1; color: white; text-decoration: none; border-radius: 25px; margin: 10px; font-weight: bold; }
                .footer { background: #343a40; color: white; padding: 20px; text-align: center; border-radius: 0 0 10px 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Your Career Growth</h1>
                    <p>Opportunities await your next move</p>
                </div>
                
                <div class="content">
                    <h2>Dear {{ employee_name }},</h2>
                    
                    <p>Your {{ tenure_years }} years with us have shown your commitment and talent. We're excited to discuss the next steps in your career journey!</p>
                    
                    <div class="career-path">
                        <h3>🎯 Opportunities Tailored for You</h3>
                        <div class="opportunities">
                            {% for role in available_roles %}
                            <div class="opportunity">
                                <h4>{{ role }}</h4>
                                <p>Based on your experience as {{ employee_role }}</p>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                    
                    <h3>🧠 Skill Development Focus Areas</h3>
                    <p>We've identified these areas that align with your career goals:</p>
                    <div class="skills-grid">
                        {% for skill in skill_development %}
                        <div class="skill-tag">{{ skill }}</div>
                        {% endfor %}
                    </div>
                    
                    <h3>💼 What We Offer</h3>
                    <ul>
                        <li><strong>Mentorship Program:</strong> Connect with senior leaders</li>
                        <li><strong>Training Budget:</strong> $2,500 annually for professional development</li>
                        <li><strong>Conference Attendance:</strong> Industry events and networking</li>
                        <li><strong>Internal Mobility:</strong> Priority consideration for new roles</li>
                        <li><strong>Leadership Track:</strong> Management development program</li>
                    </ul>
                    
                    <div style="text-align: center;">
                        <a href="{{ career_portal_url }}" class="btn">🔍 Explore Opportunities</a>
                        <a href="{{ feedback_url }}" class="btn">💬 Discuss My Goals</a>
                    </div>
                    
                    <p>Ready to take the next step? Let's schedule a career development conversation with {{ manager_name }}.</p>
                </div>
                
                <div class="footer">
                    <p>Your growth is our success<br>
                    <small>Career Development Team</small></p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_retention_career_text(self) -> str:
        """Text template for career-focused retention."""
        return """
        YOUR CAREER GROWTH - Opportunities Await Your Next Move
        
        Dear {{ employee_name }},
        
        Your {{ tenure_years }} years with us have shown your commitment and talent. We're excited to discuss the next steps in your career journey!
        
        🎯 OPPORTUNITIES TAILORED FOR YOU:
        {% for role in available_roles %}
        - {{ role }} (based on your experience as {{ employee_role }})
        {% endfor %}
        
        🧠 SKILL DEVELOPMENT FOCUS AREAS:
        {% for skill in skill_development %}
        - {{ skill }}
        {% endfor %}
        
        💼 WHAT WE OFFER:
        - Mentorship Program: Connect with senior leaders
        - Training Budget: $2,500 annually for professional development
        - Conference Attendance: Industry events and networking
        - Internal Mobility: Priority consideration for new roles
        - Leadership Track: Management development program
        
        Explore Opportunities: {{ career_portal_url }}
        Discuss My Goals: {{ feedback_url }}
        
        Ready to take the next step? Let's schedule a career development conversation with {{ manager_name }}.
        
        ---
        Your growth is our success
        Career Development Team
        """
    
    def _get_retention_wellness_html(self) -> str:
        """HTML template for wellness-focused retention."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Your Well-being Matters</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }
                .content { background: #f8f9fa; padding: 30px; }
                .wellness-score { background: white; padding: 25px; margin: 20px 0; border-radius: 10px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
                .programs-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
                .program-card { background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.1); }
                .program-icon { font-size: 40px; margin-bottom: 15px; }
                .btn { display: inline-block; padding: 15px 30px; background: #28a745; color: white; text-decoration: none; border-radius: 25px; margin: 10px; font-weight: bold; }
                .footer { background: #343a40; color: white; padding: 20px; text-align: center; border-radius: 0 0 10px 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌱 Your Well-being Matters</h1>
                    <p>Prioritizing your health and work-life balance</p>
                </div>
                
                <div class="content">
                    <h2>Dear {{ employee_name }},</h2>
                    
                    <p>We believe that your well-being is fundamental to your success and happiness, both at work and in life. That's why we're expanding our wellness initiatives with programs designed specifically for employees like you.</p>
                    
                    <div class="wellness-score">
                        <h3>📊 Your Current Work-Life Balance Score</h3>
                        <p style="font-size: 36px; font-weight: bold; color: {{ '#28a745' if work_life_score >= 3 else '#ffc107' if work_life_score >= 2 else '#dc3545' }};">
                            {{ work_life_score }}/5
                        </p>
                        <p>Let's work together to improve this!</p>
                    </div>
                    
                    <h3>🌟 Wellness Programs Available to You</h3>
                    <div class="programs-grid">
                        {% for program in wellness_programs %}
                        <div class="program-card">
                            <div class="program-icon">🧘</div>
                            <h4>{{ program }}</h4>
                            <p>Free for all employees</p>
                        </div>
                        {% endfor %}
                    </div>
                    
                    <h3>💪 Immediate Support Available</h3>
                    <ul>
                        <li><strong>Flexible Work Hours:</strong> Discuss options with {{ manager_name }}</li>
                        <li><strong>Remote Work Days:</strong> Up to 2 days per week</li>
                        <li><strong>Mental Health Days:</strong> Use your wellness PTO</li>
                        <li><strong>Employee Assistance Program:</strong> 24/7 confidential support</li>
                        <li><strong>Wellness Stipend:</strong> $500 annually for wellness activities</li>
                    </ul>
                    
                    <div style="text-align: center;">
                        <a href="{{ wellness_url }}" class="btn">🌿 Explore Wellness Programs</a>
                        <a href="{{ feedback_url }}" class="btn">💬 Request Flexibility</a>
                    </div>
                    
                    <p><strong>Remember:</strong> Your health and well-being are our priority. Don't hesitate to reach out to HR at <a href="mailto:{{ hr_contact }}">{{ hr_contact }}</a> if you need support.</p>
                </div>
                
                <div class="footer">
                    <p>Supporting your whole-person wellness<br>
                    <small>Employee Wellness Team</small></p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_retention_wellness_text(self) -> str:
        """Text template for wellness-focused retention."""
        return """
        YOUR WELL-BEING MATTERS - Prioritizing Health & Work-Life Balance
        
        Dear {{ employee_name }},
        
        We believe that your well-being is fundamental to your success and happiness, both at work and in life. That's why we're expanding our wellness initiatives with programs designed specifically for employees like you.
        
        📊 YOUR CURRENT WORK-LIFE BALANCE SCORE: {{ work_life_score }}/5
        Let's work together to improve this!
        
        🌟 WELLNESS PROGRAMS AVAILABLE:
        {% for program in wellness_programs %}
        🧘 {{ program }} (Free for all employees)
        {% endfor %}
        
        💪 IMMEDIATE SUPPORT AVAILABLE:
        - Flexible Work Hours: Discuss options with {{ manager_name }}
        - Remote Work Days: Up to 2 days per week
        - Mental Health Days: Use your wellness PTO
        - Employee Assistance Program: 24/7 confidential support
        - Wellness Stipend: $500 annually for wellness activities
        
        Explore Wellness Programs: {{ wellness_url }}
        Request Flexibility: {{ feedback_url }}
        
        REMEMBER: Your health and well-being are our priority. Don't hesitate to reach out to HR at {{ hr_contact }} if you need support.
        
        ---
        Supporting your whole-person wellness
        Employee Wellness Team
        """
    
    def get_email_statistics(self) -> Dict[str, Any]:
        """Get email sending statistics."""
        return {
            'total_sent': self.email_stats['sent'],
            'total_failed': self.email_stats['failed'],
            'success_rate': self.email_stats['sent'] / (self.email_stats['sent'] + self.email_stats['failed']) if (self.email_stats['sent'] + self.email_stats['failed']) > 0 else 0,
            'total_campaigns': self.email_stats['total_campaigns'],
            'last_sent': self.email_stats['last_sent'],
            'templates_available': len(self.templates),
            'rate_limit_per_minute': self.rate_limit,
            'emails_in_last_minute': len(self.sent_times)
        }
    
    def test_email_service(self, test_email: str) -> EmailResult:
        """Test email service configuration."""
        
        logger.info(f"🧪 Testing email service with {test_email}")
        
        # Test SMTP connection first
        if not self._check_smtp_connection():
            return EmailResult(
                success=False,
                message="SMTP connection test failed",
                recipient_email=test_email,
                timestamp=datetime.now(),
                error_details="Cannot connect to SMTP server"
            )
        
        # Send test email
        test_recipient = EmailRecipient(email=test_email, name="Test User")
        
        test_template = EmailTemplate(
            template_id='test',
            subject='🧪 Email Service Test',
            html_content="""
            <h2>Email Service Test</h2>
            <p>This is a test email from the HR Attrition Predictor email service.</p>
            <p><strong>Test successful!</strong> ✅</p>
            <p>Timestamp: {{ current_date }}</p>
            """,
            text_content="""
            Email Service Test
            
            This is a test email from the HR Attrition Predictor email service.
            
            Test successful! ✅
            
            Timestamp: {{ current_date }}
            """
        )
        
        return self._send_single_email(test_recipient, test_template)
    
    def save_templates_to_files(self):
        """Save email templates to HTML files for editing."""
        
        templates_path = self.templates_dir / "saved_templates"
        templates_path.mkdir(exist_ok=True)
        
        for template_id, template in self.templates.items():
            # Save HTML template
            html_file = templates_path / f"{template_id}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(template.html_content)
            
            # Save text template
            text_file = templates_path / f"{template_id}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(template.text_content or "")
            
            # Save metadata
            meta_file = templates_path / f"{template_id}_meta.json"
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'template_id': template.template_id,
                    'subject': template.subject,
                    'category': template.category,
                    'template_vars': template.template_vars or {}
                }, f, indent=2)
        
        logger.info(f"📁 Templates saved to {templates_path}")

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def create_smtp_config_from_env() -> SMTPConfig:
    """Create SMTP configuration from environment variables."""
    
    return SMTPConfig(
        host=os.getenv('SMTP_HOST', 'smtp.gmail.com'),
        port=int(os.getenv('SMTP_PORT', '587')),
        username=os.getenv('SMTP_USERNAME', ''),
        password=os.getenv('SMTP_PASSWORD', ''),
        use_tls=os.getenv('SMTP_USE_TLS', 'true').lower() == 'true',
        timeout=int(os.getenv('SMTP_TIMEOUT', '30'))
    )

def create_email_service() -> EmailService:
    """Create EmailService instance with environment configuration."""
    
    smtp_config = create_smtp_config_from_env()
    return EmailService(smtp_config)

# ================================================================
# EXPORT ALL CLASSES AND FUNCTIONS
# ================================================================

__all__ = [
    'EmailService',
    'SMTPConfig', 
    'EmailRecipient',
    'EmailTemplate',
    'EmailCampaign',
    'EmailResult',
    'create_smtp_config_from_env',
    'create_email_service'
]

# ================================================================
# TESTING
# ================================================================

def test_email_service():
    """Test email service functionality."""
    
    print("🧪 Testing Email Service...")
    
    # Test configuration
    smtp_config = SMTPConfig(
        host='smtp.gmail.com',
        port=587,
        username='test@example.com',
        password='test_password',
        use_tls=True
    )
    
    # Create service
    email_service = EmailService(smtp_config)
    
    print(f"✅ EmailService created with {len(email_service.templates)} templates")
    
    # Test template creation
    print("📝 Available templates:")
    for template_id in email_service.templates.keys():
        print(f"  - {template_id}")
    
    # Test statistics
    stats = email_service.get_email_statistics()
    print(f"📊 Email statistics: {stats}")
    
    print("🎉 Email service test completed!")

if __name__ == "__main__":
    test_email_service()
