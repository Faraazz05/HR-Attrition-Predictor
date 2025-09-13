"""
HR Attrition Predictor - Comprehensive Synthetic Data Generator
==============================================================
Creates realistic employee datasets with proper correlations for 
attrition prediction modeling. Generates 10,000+ employee records
with 40+ features covering personal, professional, and performance data.

Author: Mohd Faraz
Date: September 2025
"""

import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta, date
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import math
from dataclasses import dataclass

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Faker with multiple locales for diversity
fake = Faker(['en_US', 'en_CA', 'en_GB'])
Faker.seed(42)  # For reproducible results
np.random.seed(42)
random.seed(42)


@dataclass
class DataGenerationStats:
    """Statistics about the generated dataset"""
    total_employees: int
    attrition_rate: float
    avg_salary: float
    departments: Dict[str, int]
    job_roles: Dict[str, int]
    generation_time: float


class EmployeeDataGenerator:
    """
    Comprehensive synthetic HR dataset generator with realistic correlations.
    
    Creates enterprise-grade employee datasets suitable for attrition prediction,
    performance analysis, and HR analytics with proper statistical relationships.
    """
    
    def __init__(self, num_employees: int = 10000, random_state: int = 42):
        """
        Initialize the employee data generator.
        
        Args:
            num_employees: Number of employee records to generate
            random_state: Random seed for reproducibility
        """
        self.num_employees = num_employees
        self.random_state = random_state
        
        # Set seeds for reproducibility
        np.random.seed(random_state)
        random.seed(random_state)
        Faker.seed(random_state)
        
        # Initialize data storage
        self.employees_data: List[Dict[str, Any]] = []
        self.generation_stats: Optional[DataGenerationStats] = None
        
        # Define business rules and constraints
        self._define_business_rules()
        
        logger.info(f"Initialized EmployeeDataGenerator for {num_employees} employees")
    
    def _define_business_rules(self) -> None:
        """Define realistic business constraints and distributions"""
        
        # Department distribution (based on typical enterprise structure)
        self.departments = {
            'Engineering': 0.25,      # 25% - Tech company focus
            'Sales': 0.20,           # 20% - Revenue generation
            'Marketing': 0.12,       # 12% - Brand and growth
            'Operations': 0.15,      # 15% - Business operations
            'Finance': 0.08,         # 8% - Financial management
            'Human Resources': 0.05, # 5% - People management
            'Legal': 0.03,          # 3% - Legal compliance
            'Executive': 0.02,       # 2% - Leadership
            'Customer Success': 0.10 # 10% - Customer retention
        }
        
        # Job roles by department with salary ranges
        self.job_roles = {
            'Engineering': {
                'Software Engineer': (70000, 120000, 0.40),
                'Senior Software Engineer': (100000, 160000, 0.25),
                'Principal Engineer': (140000, 200000, 0.10),
                'Engineering Manager': (120000, 180000, 0.15),
                'DevOps Engineer': (80000, 140000, 0.10)
            },
            'Sales': {
                'Sales Representative': (45000, 80000, 0.40),
                'Senior Sales Rep': (60000, 100000, 0.30),
                'Sales Manager': (80000, 130000, 0.20),
                'Sales Director': (120000, 200000, 0.10)
            },
            'Marketing': {
                'Marketing Specialist': (50000, 75000, 0.35),
                'Marketing Manager': (70000, 110000, 0.30),
                'Content Creator': (45000, 70000, 0.20),
                'Marketing Director': (100000, 150000, 0.15)
            },
            'Operations': {
                'Operations Analyst': (55000, 80000, 0.30),
                'Operations Manager': (75000, 120000, 0.25),
                'Project Manager': (70000, 110000, 0.25),
                'Operations Director': (110000, 160000, 0.20)
            },
            'Finance': {
                'Financial Analyst': (55000, 85000, 0.40),
                'Senior Financial Analyst': (70000, 100000, 0.30),
                'Finance Manager': (85000, 130000, 0.20),
                'CFO': (150000, 250000, 0.10)
            },
            'Human Resources': {
                'HR Specialist': (45000, 70000, 0.40),
                'HR Manager': (65000, 95000, 0.35),
                'HR Director': (95000, 140000, 0.25)
            },
            'Legal': {
                'Legal Counsel': (90000, 140000, 0.60),
                'Senior Legal Counsel': (120000, 180000, 0.40)
            },
            'Executive': {
                'VP': (150000, 220000, 0.70),
                'CEO': (200000, 350000, 0.30)
            },
            'Customer Success': {
                'Customer Success Rep': (50000, 75000, 0.50),
                'Customer Success Manager': (70000, 105000, 0.35),
                'CS Director': (100000, 145000, 0.15)
            }
        }
        
        # Education levels with impact on salary
        self.education_levels = {
            'High School': (0.15, 0.85),     # 15% of workforce, 85% salary multiplier
            'Bachelor\'s Degree': (0.55, 1.0), # 55% of workforce, 100% salary multiplier
            'Master\'s Degree': (0.25, 1.15),  # 25% of workforce, 115% salary multiplier
            'PhD': (0.05, 1.25)              # 5% of workforce, 125% salary multiplier
        }
        
        # Performance rating distribution (bell curve)
        self.performance_distribution = {
            'Exceptional': 0.10,     # Top 10%
            'Exceeds Expectations': 0.20,  # Next 20%
            'Meets Expectations': 0.50,   # Middle 50%
            'Below Expectations': 0.15,   # Next 15%
            'Unsatisfactory': 0.05        # Bottom 5%
        }
        
        # Attrition risk factors (higher values = higher risk)
        self.attrition_factors = {
            'low_salary': 2.5,
            'poor_performance': 3.0,
            'low_satisfaction': 2.8,
            'high_overtime': 1.8,
            'long_commute': 1.5,
            'no_promotion': 1.6,
            'poor_work_life_balance': 2.2,
            'high_travel': 1.4,
            'young_age': 1.3,
            'recent_hire': 1.7
        }
    
    def generate_personal_info(self) -> Dict[str, Any]:
        """
        Generate realistic personal information for an employee.
        
        Returns:
            Dictionary containing personal demographic data
        """
        # Generate basic demographics with realistic distributions
        gender = np.random.choice(['Male', 'Female', 'Non-Binary'], p=[0.48, 0.50, 0.02])
        age = int(np.random.normal(35, 10))  # Normal distribution around 35
        age = max(22, min(65, age))  # Clamp to realistic working age
        
        # Marital status (age-correlated)
        if age < 25:
            marital_status = np.random.choice(['Single', 'Married'], p=[0.8, 0.2])
        elif age < 35:
            marital_status = np.random.choice(['Single', 'Married', 'Divorced'], p=[0.4, 0.55, 0.05])
        else:
            marital_status = np.random.choice(['Single', 'Married', 'Divorced'], p=[0.15, 0.75, 0.10])
        
        # Generate names based on gender
        if gender == 'Male':
            first_name = fake.first_name_male()
        elif gender == 'Female':
            first_name = fake.first_name_female()
        else:
            first_name = fake.first_name()
        
        last_name = fake.last_name()
        
        # Education level (age-correlated - older employees more likely to have advanced degrees)
        education_probs = [0.15, 0.55, 0.25, 0.05]  # Base probabilities
        if age > 40:
            education_probs = [0.10, 0.45, 0.35, 0.10]  # More advanced degrees
        
        education = np.random.choice(
            list(self.education_levels.keys()),
            p=education_probs
        )
        
        # Distance from home (impacts attrition)
        distance_from_home = max(1, int(np.random.exponential(15)))  # Exponential distribution
        distance_from_home = min(distance_from_home, 100)  # Cap at 100 miles
        
        return {
            'EmployeeID': f'EMP{fake.unique.random_int(min=100000, max=999999)}',
            'FirstName': first_name,
            'LastName': last_name,
            'FullName': f'{first_name} {last_name}',
            'Email': f'{first_name.lower()}.{last_name.lower()}@company.com',
            'Phone': fake.phone_number(),
            'Age': age,
            'Gender': gender,
            'MaritalStatus': marital_status,
            'Education': education,
            'DistanceFromHome': distance_from_home,
            'Address': fake.address().replace('\n', ', '),
            'EmergencyContact': fake.name(),
            'EmergencyPhone': fake.phone_number()
        }
    
    def generate_professional_info(self, personal_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate professional information with realistic correlations.
        
        Args:
            personal_info: Personal information to correlate with
            
        Returns:
            Dictionary containing professional data
        """
        age = personal_info['Age']
        education = personal_info['Education']
        
        # Select department and role
        department = np.random.choice(
            list(self.departments.keys()),
            p=list(self.departments.values())
        )
        
        # Select job role within department
        roles = self.job_roles[department]
        role_names = list(roles.keys())
        role_probs = [role[2] for role in roles.values()]
        job_role = np.random.choice(role_names, p=role_probs)
        
        # Calculate salary based on role, education, and experience
        base_salary_min, base_salary_max, _ = roles[job_role]
        education_multiplier = self.education_levels[education][1]
        
        # Experience calculation (age-based with some randomness)
        total_experience = max(0, age - 22 + np.random.randint(-2, 3))
        
        # Years at current company (subset of total experience)
        years_at_company = min(total_experience, max(0, int(np.random.exponential(4))))
        years_at_company = max(years_at_company, 1)  # At least 1 year
        
        # Years in current role (subset of years at company)
        years_in_role = min(years_at_company, max(1, int(np.random.exponential(2))))
        
        # Years since last promotion
        years_since_promotion = min(years_in_role, np.random.randint(0, 8))
        
        # Calculate salary with experience and education adjustments
        experience_multiplier = 1 + (total_experience * 0.02)  # 2% per year of experience
        salary_range = base_salary_max - base_salary_min
        salary = base_salary_min + (salary_range * np.random.random())
        monthly_income = int(salary * education_multiplier * experience_multiplier)
        
        # Job level (1-5 based on role and experience)
        if 'Senior' in job_role or 'Principal' in job_role:
            job_level = np.random.randint(3, 5)
        elif 'Manager' in job_role or 'Director' in job_role:
            job_level = np.random.randint(3, 6)
        elif job_role in ['CEO', 'VP']:
            job_level = 5
        else:
            job_level = min(4, max(1, 1 + total_experience // 3))
        
        # Hire date calculation
        hire_date = fake.date_between(
            start_date=f'-{years_at_company}y',
            end_date=f'-{max(0, years_at_company-1)}y'
        )
        
        # Manager assignment (realistic hierarchy)
        manager_id = f'MGR{fake.random_int(min=1000, max=9999)}'
        if job_level >= 4 or department == 'Executive':
            manager_id = 'CEO001'  # Senior roles report to CEO
        
        return {
            'Department': department,
            'JobRole': job_role,
            'JobLevel': job_level,
            'MonthlyIncome': monthly_income,
            'HourlyRate': int(monthly_income / 173),  # Approximate hours per month
            'DailyRate': int(monthly_income / 22),    # Approximate work days per month
            'TotalWorkingYears': total_experience,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_role,
            'YearsSinceLastPromotion': years_since_promotion,
            'HireDate': hire_date,
            'ManagerID': manager_id,
            'NumCompaniesWorked': max(1, int(np.random.exponential(2))),  # Previous companies
            'StandardHours': 40,  # Standard work week
            'StockOptionLevel': np.random.randint(0, 4)  # Stock options (0-3)
        }
    
    def generate_performance_metrics(self, professional_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate performance-related metrics with correlations.
        
        Args:
            professional_info: Professional information for correlation
            
        Returns:
            Dictionary containing performance data
        """
        job_level = professional_info['JobLevel']
        years_experience = professional_info['TotalWorkingYears']
        
        # Performance rating (correlated with experience and job level)
        perf_base_prob = [0.05, 0.15, 0.50, 0.20, 0.10]  # Base distribution
        
        # Adjust probabilities based on experience and level
        if years_experience > 10 or job_level >= 4:
            perf_base_prob = [0.02, 0.08, 0.40, 0.35, 0.15]  # Better performance for seniors
        
        performance_rating = np.random.choice(
            list(self.performance_distribution.keys()),
            p=perf_base_prob
        )
        
        # Convert to numeric score (1-5 scale)
        perf_score_map = {
            'Unsatisfactory': 1,
            'Below Expectations': 2,
            'Meets Expectations': 3,
            'Exceeds Expectations': 4,
            'Exceptional': 5
        }
        performance_score = perf_score_map[performance_rating]
        
        # Salary hike percentage (correlated with performance)
        base_hike = 3  # Base 3% hike
        performance_bonus = (performance_score - 3) * 2  # Additional based on performance
        percent_salary_hike = max(0, base_hike + performance_bonus + np.random.normal(0, 2))
        
        # Training hours (correlated with job level and performance)
        base_training = 20
        level_training = job_level * 5
        performance_training = performance_score * 3
        training_times_last_year = max(0, int(
            base_training + level_training + performance_training + np.random.normal(0, 5)
        ))
        
        # Goal achievement percentage
        goal_achievement = max(0, min(150, 
            60 + (performance_score * 15) + np.random.normal(0, 10)
        ))
        
        return {
            'PerformanceRating': performance_rating,
            'PerformanceScore': performance_score,
            'PercentSalaryHike': round(percent_salary_hike, 1),
            'TrainingTimesLastYear': training_times_last_year,
            'GoalAchievementPercent': round(goal_achievement, 1),
            'Last360ReviewScore': max(1, min(5, performance_score + np.random.normal(0, 0.5))),
            'TeamCollaborationRating': max(1, min(5, performance_score + np.random.normal(0, 0.3))),
            'InnovationScore': max(1, min(5, 3 + np.random.normal(0, 1))),
            'LeadershipPotential': max(1, min(5, performance_score + np.random.normal(0, 0.7)))
        }
    
    def generate_work_life_data(self, personal_info: Dict[str, Any], 
                              professional_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate work-life balance and satisfaction metrics.
        
        Args:
            personal_info: Personal information for correlation
            professional_info: Professional information for correlation
            
        Returns:
            Dictionary containing work-life data
        """
        age = personal_info['Age']
        distance = personal_info['DistanceFromHome']
        department = professional_info['Department']
        job_level = professional_info['JobLevel']
        
        # Overtime (correlated with department and role level)
        overtime_prob = 0.3  # Base 30% chance
        if department in ['Engineering', 'Sales']:
            overtime_prob = 0.5
        elif job_level >= 4:
            overtime_prob = 0.6  # Managers work more overtime
        
        over_time = 'Yes' if np.random.random() < overtime_prob else 'No'
        
        # Business travel frequency (role and department dependent)
        travel_options = ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']
        if department in ['Sales', 'Executive']:
            travel_probs = [0.2, 0.4, 0.4]
        elif department in ['Engineering', 'Operations']:
            travel_probs = [0.6, 0.3, 0.1]
        else:
            travel_probs = [0.5, 0.4, 0.1]
        
        business_travel = np.random.choice(travel_options, p=travel_probs)
        
        # Satisfaction scores (correlated with overtime, distance, age)
        base_satisfaction = 3  # Base satisfaction score
        
        # Adjustments based on factors
        if over_time == 'Yes':
            base_satisfaction -= 0.3
        if distance > 20:
            base_satisfaction -= 0.2
        if age < 30:
            base_satisfaction += 0.1  # Younger employees often more enthusiastic
        
        # Generate satisfaction scores with correlations
        job_satisfaction = max(1, min(4, base_satisfaction + np.random.normal(0, 0.5)))
        environment_satisfaction = max(1, min(4, job_satisfaction + np.random.normal(0, 0.3)))
        work_life_balance = max(1, min(4, base_satisfaction + np.random.normal(0, 0.4)))
        relationship_satisfaction = max(1, min(4, 3 + np.random.normal(0, 0.6)))
        
        # Job involvement (correlated with satisfaction and performance)
        job_involvement = max(1, min(4, job_satisfaction + np.random.normal(0, 0.4)))
        
        return {
            'OverTime': over_time,
            'BusinessTravel': business_travel,
            'JobSatisfaction': int(round(job_satisfaction)),
            'EnvironmentSatisfaction': int(round(environment_satisfaction)),
            'WorkLifeBalance': int(round(work_life_balance)),
            'RelationshipSatisfaction': int(round(relationship_satisfaction)),
            'JobInvolvement': int(round(job_involvement)),
            'WorkFromHomeFrequency': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often'], 
                                                    p=[0.2, 0.3, 0.35, 0.15]),
            'MentorshipProgram': 'Yes' if np.random.random() < 0.4 else 'No',
            'FlexibleSchedule': 'Yes' if np.random.random() < 0.6 else 'No'
        }
    
    def generate_leave_benefits(self, personal_info: Dict[str, Any], 
                              professional_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate leave and benefits information.
        
        Args:
            personal_info: Personal information for correlation
            professional_info: Professional information for correlation
            
        Returns:
            Dictionary containing leave and benefits data
        """
        years_at_company = professional_info['YearsAtCompany']
        job_level = professional_info['JobLevel']
        monthly_income = professional_info['MonthlyIncome']
        
        # PTO accrual based on tenure and level
        base_pto = 15  # Base 15 days
        tenure_bonus = min(10, years_at_company)  # Up to 10 extra days for tenure
        level_bonus = job_level * 2  # 2 extra days per job level
        annual_pto_days = base_pto + tenure_bonus + level_bonus
        
        # Calculate used leave (realistic patterns)
        pto_used = max(0, int(annual_pto_days * np.random.uniform(0.3, 1.2)))
        sick_leave_used = max(0, int(np.random.exponential(3)))
        
        # Benefits (correlated with job level and salary)
        health_insurance = 'Premium' if job_level >= 3 else 'Standard'
        retirement_contribution = min(15, max(3, job_level * 2 + np.random.randint(-1, 2)))
        
        # Stock options and bonuses
        annual_bonus = int(monthly_income * 12 * (0.05 + job_level * 0.02) * np.random.uniform(0.5, 1.5))
        if job_level <= 2:
            annual_bonus = max(0, annual_bonus - 5000)  # Lower levels get smaller bonuses
        
        return {
            'AnnualPTODays': annual_pto_days,
            'PTOUsed': pto_used,
            'PTOBalance': max(0, annual_pto_days - pto_used),
            'SickLeaveUsed': sick_leave_used,
            'PersonalLeaveUsed': np.random.randint(0, 5),
            'HealthInsurancePlan': health_insurance,
            'RetirementContribution': retirement_contribution,
            'AnnualBonus': annual_bonus,
            'WellnessProgramParticipation': 'Yes' if np.random.random() < 0.45 else 'No',
            'EmployeeDiscountUsage': 'High' if np.random.random() < 0.3 else 
                                   'Medium' if np.random.random() < 0.4 else 'Low'
        }
    
    def generate_attrition_target(self, employee_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate attrition target with realistic correlations to other features.
        
        Args:
            employee_data: Complete employee data for correlation analysis
            
        Returns:
            Dictionary containing attrition target and risk factors
        """
        # Initialize risk score
        risk_score = 0.0
        risk_factors = []
        
        # Age factor (younger employees more likely to leave)
        if employee_data['Age'] < 25:
            risk_score += self.attrition_factors['young_age']
            risk_factors.append('Young Employee')
        
        # Salary factor (below market rate increases risk)
        department_avg_salary = {
            'Engineering': 110000, 'Sales': 85000, 'Marketing': 75000,
            'Operations': 85000, 'Finance': 85000, 'Human Resources': 70000,
            'Legal': 130000, 'Executive': 200000, 'Customer Success': 80000
        }
        
        expected_salary = department_avg_salary.get(employee_data['Department'], 80000)
        if employee_data['MonthlyIncome'] * 12 < expected_salary * 0.8:
            risk_score += self.attrition_factors['low_salary']
            risk_factors.append('Below Market Salary')
        
        # Performance factor
        if employee_data['PerformanceScore'] <= 2:
            risk_score += self.attrition_factors['poor_performance']
            risk_factors.append('Poor Performance')
        
        # Satisfaction factors
        if employee_data['JobSatisfaction'] <= 2:
            risk_score += self.attrition_factors['low_satisfaction']
            risk_factors.append('Low Job Satisfaction')
        
        if employee_data['WorkLifeBalance'] <= 2:
            risk_score += self.attrition_factors['poor_work_life_balance']
            risk_factors.append('Poor Work-Life Balance')
        
        # Overtime factor
        if employee_data['OverTime'] == 'Yes':
            risk_score += self.attrition_factors['high_overtime']
            risk_factors.append('High Overtime')
        
        # Commute factor
        if employee_data['DistanceFromHome'] > 25:
            risk_score += self.attrition_factors['long_commute']
            risk_factors.append('Long Commute')
        
        # Promotion factor
        if employee_data['YearsSinceLastPromotion'] > 4:
            risk_score += self.attrition_factors['no_promotion']
            risk_factors.append('No Recent Promotion')
        
        # Travel factor
        if employee_data['BusinessTravel'] == 'Travel_Frequently':
            risk_score += self.attrition_factors['high_travel']
            risk_factors.append('Frequent Travel')
        
        # Tenure factor (new employees at higher risk)
        if employee_data['YearsAtCompany'] <= 1:
            risk_score += self.attrition_factors['recent_hire']
            risk_factors.append('Recent Hire')
        
        # Convert risk score to probability using sigmoid function
        attrition_probability = 1 / (1 + np.exp(-(risk_score - 4)))
        
        # Generate binary attrition decision
        attrition = 'Yes' if np.random.random() < attrition_probability else 'No'
        
        # Risk level categorization
        if attrition_probability < 0.2:
            risk_level = 'Low'
        elif attrition_probability < 0.5:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        return {
            'Attrition': attrition,
            'AttritionProbability': round(attrition_probability, 3),
            'RiskLevel': risk_level,
            'RiskScore': round(risk_score, 2),
            'RiskFactors': '; '.join(risk_factors) if risk_factors else 'None',
            'PredictedRetention': 'No' if attrition == 'Yes' else 'Yes'
        }
    
    def create_realistic_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance dataset with realistic correlations and derived features.
        
        Args:
            df: Base dataframe to enhance
            
        Returns:
            Enhanced dataframe with additional correlated features
        """
        # Create tenure-based features
        df['TenureCategory'] = pd.cut(df['YearsAtCompany'], 
                                     bins=[0, 2, 5, 10, float('inf')],
                                     labels=['New', 'Established', 'Veteran', 'Senior'])
        
        # Create salary bands
        df['SalaryBand'] = pd.cut(df['MonthlyIncome'] * 12,
                                 bins=[0, 50000, 75000, 100000, 150000, float('inf')],
                                 labels=['Entry', 'Junior', 'Mid', 'Senior', 'Executive'])
        
        # Create composite satisfaction score
        satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 
                           'WorkLifeBalance', 'RelationshipSatisfaction']
        df['OverallSatisfaction'] = df[satisfaction_cols].mean(axis=1)
        
        # Create performance index
        df['PerformanceIndex'] = (
            df['PerformanceScore'] * 0.4 +
            df['GoalAchievementPercent'] / 100 * 0.3 +
            df['Last360ReviewScore'] * 0.3
        )
        
        # Create career progression indicator
        df['CareerProgressionRate'] = df['JobLevel'] / df['YearsAtCompany'].replace(0, 1)
        
        # Create work intensity score
        df['WorkIntensityScore'] = (
            (df['OverTime'] == 'Yes').astype(int) * 2 +
            (df['BusinessTravel'] == 'Travel_Frequently').astype(int) * 1.5 +
            df['StandardHours'] / 40
        )
        
        # Age groups
        df['AgeGroup'] = pd.cut(df['Age'],
                               bins=[0, 30, 40, 50, 65],
                               labels=['Young', 'Mid-Career', 'Experienced', 'Senior'])
        
        # Employee value score (retention priority)
        df['EmployeeValueScore'] = (
            df['PerformanceScore'] * 0.3 +
            df['OverallSatisfaction'] * 0.2 +
            df['YearsAtCompany'] * 0.1 +
            (df['MonthlyIncome'] / df['MonthlyIncome'].max()) * 5 * 0.2 +
            df['TrainingTimesLastYear'] / 50 * 0.2
        )
        
        return df
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate complete synthetic employee dataset.
        
        Returns:
            Complete pandas DataFrame with all employee data
        """
        start_time = datetime.now()
        logger.info(f"Starting generation of {self.num_employees} employee records...")
        
        employees_data = []
        
        for i in range(self.num_employees):
            if i % 1000 == 0:
                logger.info(f"Generated {i}/{self.num_employees} employees...")
            
            # Generate data in order with correlations
            personal_info = self.generate_personal_info()
            professional_info = self.generate_professional_info(personal_info)
            performance_metrics = self.generate_performance_metrics(professional_info)
            work_life_data = self.generate_work_life_data(personal_info, professional_info)
            leave_benefits = self.generate_leave_benefits(personal_info, professional_info)
            
            # Combine all data
            employee_record = {
                **personal_info,
                **professional_info,
                **performance_metrics,
                **work_life_data,
                **leave_benefits
            }
            
            # Generate attrition target based on all factors
            attrition_data = self.generate_attrition_target(employee_record)
            employee_record.update(attrition_data)
            
            employees_data.append(employee_record)
        
        # Create DataFrame
        df = pd.DataFrame(employees_data)
        
        # Add realistic correlations and derived features
        df = self.create_realistic_correlations(df)
        
        # Generate statistics
        generation_time = (datetime.now() - start_time).total_seconds()
        attrition_rate = (df['Attrition'] == 'Yes').mean()
        avg_salary = (df['MonthlyIncome'] * 12).mean()
        
        self.generation_stats = DataGenerationStats(
            total_employees=len(df),
            attrition_rate=attrition_rate,
            avg_salary=avg_salary,
            departments=df['Department'].value_counts().to_dict(),
            job_roles=df['JobRole'].value_counts().to_dict(),
            generation_time=generation_time
        )
        
        logger.info(f"Dataset generation completed in {generation_time:.2f} seconds")
        logger.info(f"Generated {len(df)} employees with {len(df.columns)} features")
        logger.info(f"Attrition rate: {attrition_rate:.2%}")
        logger.info(f"Average annual salary: ${avg_salary:,.0f}")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, file_path: Optional[str] = None) -> str:
        """
        Save dataset to CSV file with metadata.
        
        Args:
            df: DataFrame to save
            file_path: Optional custom file path
            
        Returns:
            Path where file was saved
        """
        if file_path is None:
            # Use default path structure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"C:/Users/Faraz/Documents/hr_attrition_predictor/data/synthetic/hr_employees_{timestamp}.csv"
        
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save main dataset
        df.to_csv(file_path, index=False)
        
        # Save metadata
        metadata_path = file_path.replace('.csv', '_metadata.json')
        if self.generation_stats:
            import json
            metadata = {
                'generation_date': datetime.now().isoformat(),
                'total_employees': self.generation_stats.total_employees,
                'total_features': len(df.columns),
                'attrition_rate': self.generation_stats.attrition_rate,
                'avg_salary': self.generation_stats.avg_salary,
                'generation_time_seconds': self.generation_stats.generation_time,
                'departments': self.generation_stats.departments,
                'feature_list': df.columns.tolist()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset saved to: {file_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return file_path
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data summary for validation.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary containing data summary statistics
        """
        summary = {
            'basic_info': {
                'total_records': len(df),
                'total_features': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'missing_values': df.isnull().sum().sum()
            },
            'target_distribution': {
                'attrition_rate': (df['Attrition'] == 'Yes').mean(),
                'attrition_count': (df['Attrition'] == 'Yes').sum(),
                'retention_count': (df['Attrition'] == 'No').sum()
            },
            'demographic_distribution': {
                'age_stats': {
                    'mean': df['Age'].mean(),
                    'min': df['Age'].min(),
                    'max': df['Age'].max(),
                    'std': df['Age'].std()
                },
                'gender_distribution': df['Gender'].value_counts().to_dict(),
                'education_distribution': df['Education'].value_counts().to_dict()
            },
            'professional_distribution': {
                'department_counts': df['Department'].value_counts().to_dict(),
                'job_level_distribution': df['JobLevel'].value_counts().to_dict(),
                'salary_stats': {
                    'mean_annual': (df['MonthlyIncome'] * 12).mean(),
                    'median_annual': (df['MonthlyIncome'] * 12).median(),
                    'min_annual': (df['MonthlyIncome'] * 12).min(),
                    'max_annual': (df['MonthlyIncome'] * 12).max()
                }
            },
            'satisfaction_stats': {
                'job_satisfaction_mean': df['JobSatisfaction'].mean(),
                'work_life_balance_mean': df['WorkLifeBalance'].mean(),
                'overall_satisfaction_mean': df['OverallSatisfaction'].mean()
            }
        }
        
        return summary


def test_data_generator():
    """Test function to validate data generation"""
    print("🚀 Testing HR Employee Data Generator...")
    
    # Generate small test dataset
    generator = EmployeeDataGenerator(num_employees=100)
    test_df = generator.generate_dataset()
    
    # Basic validation
    assert len(test_df) == 100, "Dataset size mismatch"
    assert len(test_df.columns) >= 40, "Insufficient features"
    assert test_df['Attrition'].isin(['Yes', 'No']).all(), "Invalid attrition values"
    assert test_df['Age'].between(22, 65).all(), "Invalid age range"
    assert test_df['MonthlyIncome'].gt(0).all(), "Invalid salary values"
    
    # Get summary
    summary = generator.get_data_summary(test_df)
    
    print(f"✅ Test passed! Generated {len(test_df)} employees with {len(test_df.columns)} features")
    print(f"📊 Attrition rate: {summary['target_distribution']['attrition_rate']:.2%}")
    print(f"💰 Average salary: ${summary['professional_distribution']['salary_stats']['mean_annual']:,.0f}")
    print(f"🏢 Departments: {len(summary['professional_distribution']['department_counts'])}")
    
    return test_df


if __name__ == "__main__":
    # Run test
    test_df = test_data_generator()
    
    # Generate full dataset
    print("\n🚀 Generating full dataset...")
    generator = EmployeeDataGenerator(num_employees=10000)
    full_df = generator.generate_dataset()
    
    # Save dataset
    file_path = generator.save_dataset(full_df, "data/synthetic/hr_employees.csv")
    
    # Display summary
    summary = generator.get_data_summary(full_df)
    print(f"\n📈 Final Dataset Summary:")
    print(f"Total Employees: {summary['basic_info']['total_records']:,}")
    print(f"Total Features: {summary['basic_info']['total_features']}")
    print(f"Attrition Rate: {summary['target_distribution']['attrition_rate']:.2%}")
    print(f"Average Age: {summary['demographic_distribution']['age_stats']['mean']:.1f}")
    print(f"Dataset saved to: {file_path}")
