from datetime import datetime, timedelta

start_date = datetime(2025, 3, 3) # week of first meeting 
num_weeks = 25  # 23 weeks 4 days (deadline 15th of August 2025)

for i in range(num_weeks):
    date = start_date + timedelta(weeks=i)
    formatted = date.strftime("%B %#d") 
    print(f"\\gantttitle{{{formatted} (W{i+1})}}{{1}}")