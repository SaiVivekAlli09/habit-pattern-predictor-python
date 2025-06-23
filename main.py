import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class HabitDatabase:
    """Database manager for habit tracking data"""
    
    def __init__(self, db_path="habit_tracker.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main habits table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS habits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                sleep_hours REAL,
                exercise_minutes INTEGER,
                steps_count INTEGER,
                mood_rating INTEGER,
                productivity_score INTEGER,
                work_hours REAL,
                stress_level INTEGER,
                caffeine_intake INTEGER,
                screen_time_hours REAL,
                social_interactions INTEGER,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Daily targets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_targets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                target_steps INTEGER,
                target_exercise_minutes INTEGER,
                target_sleep_hours REAL,
                target_productivity_score INTEGER,
                achieved_steps INTEGER DEFAULT 0,
                achieved_exercise_minutes INTEGER DEFAULT 0,
                achieved_sleep_hours REAL DEFAULT 0,
                achieved_productivity_score INTEGER DEFAULT 0
            )
        ''')
        
        # Activity predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                activity_type TEXT NOT NULL,
                predicted_optimal_time TEXT,
                predicted_performance_score REAL,
                actual_performance_score REAL DEFAULT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_habit_entry(self, date, sleep_hours, exercise_minutes, steps_count, 
                       mood_rating, productivity_score, work_hours=8, 
                       stress_level=5, caffeine_intake=1, screen_time_hours=6, 
                       social_interactions=3):
        """Add a new habit entry to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO habits (date, sleep_hours, exercise_minutes, steps_count, 
                              mood_rating, productivity_score, work_hours, stress_level, 
                              caffeine_intake, screen_time_hours, social_interactions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (date, sleep_hours, exercise_minutes, steps_count, mood_rating, 
              productivity_score, work_hours, stress_level, caffeine_intake, 
              screen_time_hours, social_interactions))
        
        conn.commit()
        conn.close()
    
    def set_daily_targets(self, date, target_steps, target_exercise_minutes, 
                         target_sleep_hours, target_productivity_score):
        """Set daily targets"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO daily_targets 
            (date, target_steps, target_exercise_minutes, target_sleep_hours, target_productivity_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (date, target_steps, target_exercise_minutes, target_sleep_hours, target_productivity_score))
        
        conn.commit()
        conn.close()
    
    def get_habit_data(self, days_back=90):
        """Retrieve habit data for analysis"""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT * FROM habits 
            WHERE date >= date('now', '-{} days')
            ORDER BY date DESC
        '''.format(days_back)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_targets_data(self, days_back=30):
        """Retrieve targets data"""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT * FROM daily_targets 
            WHERE date >= date('now', '-{} days')
            ORDER BY date DESC
        '''.format(days_back)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

class HabitTracker:
    """Main habit tracking and input system"""
    
    def __init__(self):
        self.db = HabitDatabase()
    
    def log_daily_habits(self):
        """Interactive habit logging"""
        print("=== Daily Habit Tracker ===")
        date = input("Enter date (YYYY-MM-DD) or press Enter for today: ")
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            sleep_hours = float(input("Hours of sleep: "))
            exercise_minutes = int(input("Exercise minutes: "))
            steps_count = int(input("Steps walked: "))
            mood_rating = int(input("Mood rating (1-10): "))
            productivity_score = int(input("Productivity score (1-10): "))
            
            # Optional metrics
            work_hours = float(input("Work hours (default 8): ") or "8")
            stress_level = int(input("Stress level (1-10, default 5): ") or "5")
            caffeine_intake = int(input("Caffeine servings (default 1): ") or "1")
            screen_time = float(input("Screen time hours (default 6): ") or "6")
            social_interactions = int(input("Social interactions count (default 3): ") or "3")
            
            self.db.add_habit_entry(
                date, sleep_hours, exercise_minutes, steps_count,
                mood_rating, productivity_score, work_hours,
                stress_level, caffeine_intake, screen_time, social_interactions
            )
            
            print("✅ Habit data logged successfully!")
            
        except ValueError:
            print("❌ Invalid input. Please enter numeric values.")
    
    def set_targets(self):
        """Set daily targets"""
        print("=== Set Daily Targets ===")
        date = input("Enter date (YYYY-MM-DD) or press Enter for today: ")
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            target_steps = int(input("Target steps: "))
            target_exercise = int(input("Target exercise minutes: "))
            target_sleep = float(input("Target sleep hours: "))
            target_productivity = int(input("Target productivity score (1-10): "))
            
            self.db.set_daily_targets(date, target_steps, target_exercise, 
                                    target_sleep, target_productivity)
            print("🎯 Daily targets set successfully!")
            
        except ValueError:
            print("❌ Invalid input. Please enter numeric values.")

class PatternAnalyzer:
    """Analyze patterns in habit data"""
    
    def __init__(self, db):
        self.db = db
        self.scaler = StandardScaler()
    
    def analyze_correlations(self):
        """Analyze correlations between different metrics"""
        df = self.db.get_habit_data()
        if df.empty:
            print("No data available for analysis")
            return
        
        # Select numeric columns for correlation
        numeric_cols = ['sleep_hours', 'exercise_minutes', 'steps_count', 
                       'mood_rating', 'productivity_score', 'work_hours', 
                       'stress_level', 'caffeine_intake', 'screen_time_hours', 
                       'social_interactions']
        
        correlation_matrix = df[numeric_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Habit Correlations Heatmap')
        plt.tight_layout()
        plt.savefig('habit_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_matrix
    
    def find_optimal_patterns(self):
        """Find patterns for optimal performance"""
        df = self.db.get_habit_data()
        if df.empty:
            return {}
        
        # Convert date to datetime for day of week analysis
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.day_name()
        df['hour'] = df['date'].dt.hour
        
        patterns = {}
        
        # Best days for high productivity
        productivity_by_day = df.groupby('day_of_week')['productivity_score'].mean()
        patterns['best_productivity_days'] = productivity_by_day.nlargest(3).to_dict()
        
        # Optimal sleep range for best mood
        df['sleep_range'] = pd.cut(df['sleep_hours'], bins=[0, 6, 7, 8, 9, 12], 
                                  labels=['<6h', '6-7h', '7-8h', '8-9h', '>9h'])
        mood_by_sleep = df.groupby('sleep_range')['mood_rating'].mean()
        patterns['optimal_sleep_for_mood'] = mood_by_sleep.to_dict()
        
        # Exercise impact on mood and productivity
        df['exercise_range'] = pd.cut(df['exercise_minutes'], 
                                    bins=[0, 30, 60, 90, 180],
                                    labels=['0-30min', '30-60min', '60-90min', '>90min'])
        exercise_impact = df.groupby('exercise_range')[['mood_rating', 'productivity_score']].mean()
        patterns['exercise_impact'] = exercise_impact.to_dict()
        
        return patterns
    
    def weekly_trend_analysis(self):
        """Analyze weekly trends"""
        df = self.db.get_habit_data()
        if df.empty:
            return
        
        df['date'] = pd.to_datetime(df['date'])
        df['week'] = df['date'].dt.isocalendar().week
        
        # Weekly averages
        weekly_stats = df.groupby('week').agg({
            'sleep_hours': 'mean',
            'exercise_minutes': 'mean',
            'steps_count': 'mean',
            'mood_rating': 'mean',
            'productivity_score': 'mean'
        }).round(2)
        
        # Plot weekly trends
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        metrics = ['sleep_hours', 'exercise_minutes', 'steps_count', 
                  'mood_rating', 'productivity_score']
        
        for i, metric in enumerate(metrics):
            row, col = i // 3, i % 3
            weekly_stats[metric].plot(kind='line', ax=axes[row, col], marker='o')
            axes[row, col].set_title(f'Weekly {metric.replace("_", " ").title()}')
            axes[row, col].grid(True, alpha=0.3)
        
        # Remove empty subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('weekly_trends.png', dpi=300, bbox_inches='tight')
        plt.show()

class ActivityPredictor:
    """Predict optimal times for activities using machine learning"""
    
    def __init__(self, db):
        self.db = db
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'sleep_hours', 'exercise_minutes', 'steps_count', 'work_hours',
            'stress_level', 'caffeine_intake', 'screen_time_hours', 
            'social_interactions', 'day_of_week_encoded', 'hour'
        ]
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['hour'] = df['date'].dt.hour
        df['day_of_week_encoded'] = df['day_of_week']
        
        # Create derived features
        df['sleep_deficit'] = 8 - df['sleep_hours']  # Assuming 8h is optimal
        df['exercise_intensity'] = df['exercise_minutes'] / 60  # Convert to hours
        df['step_goal_ratio'] = df['steps_count'] / 10000  # Assuming 10k step goal
        
        return df
    
    def train_productivity_model(self):
        """Train model to predict productivity based on habits"""
        df = self.db.get_habit_data()
        if len(df) < 10:
            print("Need at least 10 data points to train the model")
            return None
        
        df = self.prepare_features(df)
        
        # Features for prediction
        feature_cols = ['sleep_hours', 'exercise_minutes', 'work_hours', 
                       'stress_level', 'caffeine_intake', 'day_of_week_encoded']
        
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df['productivity_score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Productivity Model - MSE: {mse:.2f}, R²: {r2:.2f}")
        
        # Save model and scaler
        self.models['productivity'] = model
        self.scalers['productivity'] = scaler
        
        # Save to disk
        joblib.dump(model, 'productivity_model.pkl')
        joblib.dump(scaler, 'productivity_scaler.pkl')
        
        return model, scaler
    
    def train_mood_model(self):
        """Train model to predict mood"""
        df = self.db.get_habit_data()
        if len(df) < 10:
            print("Need at least 10 data points to train the model")
            return None
        
        df = self.prepare_features(df)
        
        feature_cols = ['sleep_hours', 'exercise_minutes', 'steps_count',
                       'stress_level', 'social_interactions', 'day_of_week_encoded']
        
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df['mood_rating']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mood Model - MSE: {mse:.2f}, R²: {r2:.2f}")
        
        self.models['mood'] = model
        self.scalers['mood'] = scaler
        
        joblib.dump(model, 'mood_model.pkl')
        joblib.dump(scaler, 'mood_scaler.pkl')
        
        return model, scaler
    
    def predict_optimal_times(self, target_date=None):
        """Predict optimal times for activities"""
        if not target_date:
            target_date = datetime.now().date()
        
        predictions = {}
        
        # Load models if not in memory
        try:
            if 'productivity' not in self.models:
                self.models['productivity'] = joblib.load('productivity_model.pkl')
                self.scalers['productivity'] = joblib.load('productivity_scaler.pkl')
        except FileNotFoundError:
            print("Training productivity model...")
            self.train_productivity_model()
        
        # Simulate different scenarios for the target date
        scenarios = []
        for hour in range(6, 23):  # 6 AM to 10 PM
            for sleep_hours in [6, 7, 8, 9]:
                for exercise_mins in [0, 30, 60]:
                    scenario = {
                        'sleep_hours': sleep_hours,
                        'exercise_minutes': exercise_mins,
                        'work_hours': 8,
                        'stress_level': 5,
                        'caffeine_intake': 1,
                        'day_of_week_encoded': target_date.weekday(),
                        'hour': hour
                    }
                    scenarios.append(scenario)
        
        # Convert to DataFrame
        scenario_df = pd.DataFrame(scenarios)
        
        # Predict productivity for each scenario
        if 'productivity' in self.models:
            feature_cols = ['sleep_hours', 'exercise_minutes', 'work_hours', 
                           'stress_level', 'caffeine_intake', 'day_of_week_encoded']
            X_scenarios = scenario_df[feature_cols]
            X_scaled = self.scalers['productivity'].transform(X_scenarios)
            productivity_predictions = self.models['productivity'].predict(X_scaled)
            scenario_df['predicted_productivity'] = productivity_predictions
            
            # Find optimal conditions
            best_scenario = scenario_df.loc[scenario_df['predicted_productivity'].idxmax()]
            
            predictions['optimal_work_time'] = {
                'recommended_sleep': best_scenario['sleep_hours'],
                'recommended_exercise': best_scenario['exercise_minutes'],
                'predicted_productivity': best_scenario['predicted_productivity'],
                'optimal_hour': best_scenario['hour']
            }
        
        return predictions
    
    def get_personalized_recommendations(self):
        """Get personalized recommendations based on historical data"""
        df = self.db.get_habit_data()
        if df.empty:
            return []
        
        recommendations = []
        
        # Analyze personal patterns
        avg_sleep = df['sleep_hours'].mean()
        avg_exercise = df['exercise_minutes'].mean()
        avg_productivity = df['productivity_score'].mean()
        avg_mood = df['mood_rating'].mean()
        
        # Sleep recommendations
        if avg_sleep < 7:
            recommendations.append({
                'category': 'Sleep',
                'recommendation': f'Increase sleep to 7-8 hours. Current average: {avg_sleep:.1f}h',
                'priority': 'high',
                'expected_benefit': 'Improved mood and productivity'
            })
        elif avg_sleep > 9:
            recommendations.append({
                'category': 'Sleep',
                'recommendation': f'Consider reducing sleep slightly. Current average: {avg_sleep:.1f}h',
                'priority': 'medium',
                'expected_benefit': 'Better sleep quality'
            })
        
        # Exercise recommendations
        if avg_exercise < 30:
            recommendations.append({
                'category': 'Exercise',
                'recommendation': f'Increase daily exercise. Current average: {avg_exercise:.0f} min',
                'priority': 'high',
                'expected_benefit': 'Better mood and energy levels'
            })
        
        # Productivity analysis
        best_prod_days = df.groupby(df['date'].str[:10])['productivity_score'].mean()
        if len(best_prod_days) > 0:
            best_day = best_prod_days.idxmax()
            best_day_data = df[df['date'].str.startswith(best_day)]
            if not best_day_data.empty:
                recommendations.append({
                    'category': 'Productivity Pattern',
                    'recommendation': f'Replicate conditions from {best_day}: '
                                    f'{best_day_data.iloc[0]["sleep_hours"]:.1f}h sleep, '
                                    f'{best_day_data.iloc[0]["exercise_minutes"]}min exercise',
                    'priority': 'medium',
                    'expected_benefit': 'Higher productivity scores'
                })
        
        return recommendations

class ReportGenerator:
    """Generate comprehensive reports and insights"""
    
    def __init__(self, db):
        self.db = db
    
    def generate_weekly_report(self):
        """Generate a comprehensive weekly report"""
        df = self.db.get_habit_data(days_back=7)
        if df.empty:
            print("No data available for the past week")
            return
        
        report = {
            'period': 'Past 7 Days',
            'summary': {},
            'achievements': [],
            'areas_for_improvement': [],
            'insights': []
        }
        
        # Summary statistics
        report['summary'] = {
            'avg_sleep': df['sleep_hours'].mean(),
            'total_exercise': df['exercise_minutes'].sum(),
            'avg_steps': df['steps_count'].mean(),
            'avg_mood': df['mood_rating'].mean(),
            'avg_productivity': df['productivity_score'].mean()
        }
        
        # Check achievements
        if report['summary']['avg_sleep'] >= 7:
            report['achievements'].append("Maintained healthy sleep schedule")
        
        if report['summary']['total_exercise'] >= 150:
            report['achievements'].append("Met weekly exercise recommendation (150+ minutes)")
        
        if report['summary']['avg_mood'] >= 7:
            report['achievements'].append("Maintained positive mood")
        
        # Areas for improvement
        if report['summary']['avg_sleep'] < 7:
            report['areas_for_improvement'].append("Increase sleep duration")
        
        if report['summary']['total_exercise'] < 150:
            report['areas_for_improvement'].append("Increase weekly exercise")
        
        if report['summary']['avg_productivity'] < 6:
            report['areas_for_improvement'].append("Focus on productivity improvement")
        
        # Generate insights
        correlation = df[['sleep_hours', 'mood_rating']].corr().iloc[0, 1]
        if abs(correlation) > 0.3:
            report['insights'].append(f"Sleep and mood correlation: {correlation:.2f}")
        
        return report
    
    def print_report(self, report):
        """Print formatted report"""
        print("\n" + "="*50)
        print(f"HABIT TRACKER REPORT - {report['period']}")
        print("="*50)
        
        print("\n📊 SUMMARY:")
        print(f"Average Sleep: {report['summary']['avg_sleep']:.1f} hours")
        print(f"Total Exercise: {report['summary']['total_exercise']:.0f} minutes")
        print(f"Average Steps: {report['summary']['avg_steps']:.0f}")
        print(f"Average Mood: {report['summary']['avg_mood']:.1f}/10")
        print(f"Average Productivity: {report['summary']['avg_productivity']:.1f}/10")
        
        if report['achievements']:
            print("\n🏆 ACHIEVEMENTS:")
            for achievement in report['achievements']:
                print(f"✅ {achievement}")
        
        if report['areas_for_improvement']:
            print("\n🎯 AREAS FOR IMPROVEMENT:")
            for area in report['areas_for_improvement']:
                print(f"📈 {area}")
        
        if report['insights']:
            print("\n💡 INSIGHTS:")
            for insight in report['insights']:
                print(f"🔍 {insight}")
        
        print("\n" + "="*50)

class HabitPatternPredictor:
    """Main application class that orchestrates all components"""
    
    def __init__(self):
        self.db = HabitDatabase()
        self.tracker = HabitTracker()
        self.analyzer = PatternAnalyzer(self.db)
        self.predictor = ActivityPredictor(self.db)
        self.reporter = ReportGenerator(self.db)
    
    def run_demo_data(self):
        """Generate demo data for testing"""
        print("Generating demo data...")
        
        # Generate 30 days of sample data
        start_date = datetime.now() - timedelta(days=30)
        
        for i in range(30):
            current_date = start_date + timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Generate realistic sample data with some patterns
            sleep_hours = np.random.normal(7.5, 1.0)
            sleep_hours = max(5, min(10, sleep_hours))  # Clamp between 5-10
            
            # Exercise varies by day of week (more on weekends)
            if current_date.weekday() >= 5:  # Weekend
                exercise_minutes = int(np.random.normal(45, 15))
            else:
                exercise_minutes = int(np.random.normal(25, 10))
            exercise_minutes = max(0, exercise_minutes)
            
            steps_count = int(np.random.normal(8000, 2000))
            steps_count = max(2000, steps_count)
            
            # Mood correlates with sleep and exercise
            mood_base = (sleep_hours - 5) * 1.5 + (exercise_minutes / 60) * 2
            mood_rating = int(np.random.normal(mood_base + 4, 1.5))
            mood_rating = max(1, min(10, mood_rating))
            
            # Productivity correlates with sleep and mood
            prod_base = (sleep_hours - 5) * 1.2 + mood_rating * 0.5
            productivity_score = int(np.random.normal(prod_base + 2, 1.2))
            productivity_score = max(1, min(10, productivity_score))
            
            work_hours = np.random.normal(8, 1)
            work_hours = max(6, min(12, work_hours))
            
            stress_level = int(np.random.normal(5, 2))
            stress_level = max(1, min(10, stress_level))
            
            caffeine_intake = int(np.random.normal(2, 1))
            caffeine_intake = max(0, min(5, caffeine_intake))
            
            screen_time = np.random.normal(6, 2)
            screen_time = max(2, min(12, screen_time))
            
            social_interactions = int(np.random.normal(4, 2))
            social_interactions = max(0, min(10, social_interactions))
            
            self.db.add_habit_entry(
                date_str, sleep_hours, exercise_minutes, steps_count,
                mood_rating, productivity_score, work_hours,
                stress_level, caffeine_intake, screen_time, social_interactions
            )
            
            # Set some targets
            if i % 5 == 0:  # Every 5 days
                self.db.set_daily_targets(date_str, 10000, 30, 8.0, 8)
        
        print("✅ Demo data generated successfully!")
    
    def main_menu(self):
        """Main application menu"""
        while True:
            print("\n" + "="*40)
            print("🎯 HABIT PATTERN PREDICTOR")
            print("="*40)
            print("1. Log Daily Habits")
            print("2. Set Daily Targets")
            print("3. View Analytics Dashboard")
            print("4. Get Predictions & Recommendations")
            print("5. Generate Reports")
            print("6. Train Prediction Models")
            print("7. Generate Demo Data")
            print("0. Exit")
            
            choice = input("\nSelect option (0-7): ")
            
            try:
                if choice == '1':
                    self.tracker.log_daily_habits()
                elif choice == '2':
                    self.tracker.set_targets()
                elif choice == '3':
                    self.show_analytics()
                elif choice == '4':
                    self.show_predictions()
                elif choice == '5':
                    self.show_reports()
                elif choice == '6':
                    self.train_models()
                elif choice == '7':
                    self.run_demo_data()
                elif choice == '0':
                    print("Thanks for using Habit Pattern Predictor! 👋")
                    break
                else:
                    print("Invalid option. Please try again.")
            except Exception as e:
                print(f"An error occurred: {e}")
    
    def show_analytics(self):
        """Show analytics dashboard"""
        print("\n📊 ANALYTICS DASHBOARD")
        print("-" * 30)
        
        # Correlation analysis
        print("Generating correlation analysis...")
        correlations = self.analyzer.analyze_correlations()
        
        # Pattern analysis
        print("\nFinding optimal patterns...")
        patterns = self.analyzer.find_optimal_patterns()
        
        if patterns:
            print("\n🏆 OPTIMAL PATTERNS FOUND:")
            if 'best_productivity_days' in patterns:
                print("\nBest days for productivity:")
                for day, score in patterns['best_productivity_days'].items():
                    print(f"  {day}: {score:.1f}/10")
            
            if 'optimal_sleep_for_mood' in patterns:
                print("\nOptimal sleep ranges for mood:")
                for sleep_range, mood in patterns['optimal_sleep_for_mood'].items():
                    print(
