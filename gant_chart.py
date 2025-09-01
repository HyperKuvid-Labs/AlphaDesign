import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.patches import Patch
import seaborn as sns

class ProjectTimelineVisualizer:
    """
    Professional project timeline visualization tool for F1 Front Wing Optimization.
    
    Generates Gantt charts and project statistics for aerodynamic engineering workflows.
    """
    
    def __init__(self, project_title="F1 Front Wing Optimization Project"):
        self.project_title = project_title
        self.status_colors = {
            'Planned': '#1f77b4',      # Professional blue
            'In Progress': '#ff7f0e',   # Orange
            'Completed': '#2ca02c',     # Green
            'Delayed': '#d62728'        # Red
        }
        
    def create_timeline_data(self):
        """Define project tasks with professional naming convention."""
        
        tasks = [
            ("Bezier Curve Surface Representation", "2025-09-02", "2025-09-07", "Planned"),
            ("Neural Network Surrogate Model Development", "2025-09-08", "2025-09-14", "Planned"),
            ("Advanced Surface Quality Control Systems", "2025-09-15", "2025-09-19", "Planned"),
            ("Multi-Element Collision Detection & Spacing", "2025-09-20", "2025-09-24", "Planned"),
            ("Controlled Flap Angle Progression Algorithm", "2025-09-25", "2025-09-28", "Planned"),
            ("Incremental Element Offset Optimization", "2025-09-29", "2025-10-02", "Planned"),
            ("Surface Quality Enhancement & Validation", "2025-10-03", "2025-10-06", "Planned"),
            ("Final System Validation & Documentation", "2025-10-07", "2025-10-10", "Planned")
        ]
        
        return pd.DataFrame(tasks, columns=['Task', 'Start_Date', 'End_Date', 'Status'])
    
    def preprocess_data(self, df):
        """Convert dates and calculate durations."""
        
        df['Start_Date'] = pd.to_datetime(df['Start_Date'])
        df['End_Date'] = pd.to_datetime(df['End_Date'])
        df['Duration_Days'] = (df['End_Date'] - df['Start_Date']).dt.days + 1
        df['Task_ID'] = range(len(df))
        
        return df
    
    def create_gantt_chart(self, df, current_date=None):
        """Generate professional Gantt chart visualization."""
        
        # Set professional styling
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Create horizontal bars
        for idx, row in df.iterrows():
            ax.barh(idx, row['Duration_Days'], 
                   left=row['Start_Date'], 
                   height=0.7,
                   color=self.status_colors[row['Status']], 
                   alpha=0.8, 
                   edgecolor='white', 
                   linewidth=1.2)
            
            # Add task duration labels
            mid_date = row['Start_Date'] + pd.Timedelta(days=row['Duration_Days']/2)
            ax.text(mid_date, idx, f"{row['Duration_Days']}d", 
                   ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Configure y-axis
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['Task'], fontsize=11)
        ax.invert_yaxis()
        
        # Configure x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        plt.xticks(rotation=45)
        
        # Add current date marker if provided
        if current_date:
            ax.axvline(x=current_date, color='red', linestyle='--', 
                      linewidth=2, alpha=0.8, label=f'Current Date: {current_date.strftime("%b %d, %Y")}')
        
        # Styling and labels
        ax.set_xlabel('Project Timeline (2025)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Work Breakdown Structure', fontsize=13, fontweight='bold')
        ax.set_title(f'{self.project_title}\nPhase 2: Advanced Optimization Implementation', 
                    fontsize=16, fontweight='bold', pad=25)
        
        # Professional grid
        ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.grid(True, axis='y', alpha=0.1, linestyle='-', linewidth=0.5)
        
        # Legend
        legend_elements = [Patch(facecolor=color, label=status) 
                          for status, color in self.status_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, shadow=True)
        
        # Professional layout
        plt.tight_layout()
        return fig, ax
    
    def generate_project_metrics(self, df):
        """Calculate and display project metrics."""
        
        metrics = {
            'Total Project Duration': (df['End_Date'].max() - df['Start_Date'].min()).days + 1,
            'Number of Tasks': len(df),
            'Average Task Duration': round(df['Duration_Days'].mean(), 1),
            'Project Start Date': df['Start_Date'].min().strftime('%B %d, %Y'),
            'Project End Date': df['End_Date'].max().strftime('%B %d, %Y'),
            'Critical Path Length': df['Duration_Days'].sum(),
            'Task Distribution': df['Status'].value_counts().to_dict()
        }
        
        return metrics
    
    def print_project_summary(self, metrics):
        """Display professional project summary."""
        
        print("=" * 60)
        print("F1 FRONT WING OPTIMIZATION PROJECT SUMMARY")
        print("=" * 60)
        print(f"Project Duration:        {metrics['Total Project Duration']} days")
        print(f"Project Start:           {metrics['Project Start Date']}")
        print(f"Project End:             {metrics['Project End Date']}")
        print(f"Total Work Packages:     {metrics['Number of Tasks']}")
        print(f"Average Task Duration:   {metrics['Average Task Duration']} days")
        print(f"Critical Path Length:    {metrics['Critical Path Length']} days")
        print("\nTask Status Distribution:")
        for status, count in metrics['Task Distribution'].items():
            print(f"  • {status}: {count} tasks")
        print("=" * 60)
    
    def run_complete_analysis(self, current_date_str="2025-09-01"):
        """Execute complete project timeline analysis."""
        
        # Create and process data
        df = self.create_timeline_data()
        df = self.preprocess_data(df)
        
        # Generate visualizations
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
        fig, ax = self.create_gantt_chart(df, current_date)
        
        # Calculate metrics
        metrics = self.generate_project_metrics(df)
        self.print_project_summary(metrics)
        
        # Display chart
        plt.savefig('gant_chart.png')
        plt.show()
        
        return df, metrics, fig

# Usage Example
if __name__ == "__main__":
    # Initialize professional timeline visualizer
    visualizer = ProjectTimelineVisualizer("F1 Aerodynamics Engineering Project")
    
    # Generate complete project analysis
    timeline_data, project_metrics, chart = visualizer.run_complete_analysis("2025-09-02")
    
    # Optional: Save chart
    # chart.savefig('f1_project_timeline.png', dpi=300, bbox_inches='tight')
