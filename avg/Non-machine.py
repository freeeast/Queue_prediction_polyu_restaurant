import math
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_wait_time(arrival_rate, service_rate, people_ahead, counters):
    counters = int(counters)
    total_service_capacity = service_rate * counters

    if total_service_capacity < arrival_rate or counters==1:
        estimated_wait_time = people_ahead / total_service_capacity
    else:
        traffic_intensity = arrival_rate / total_service_capacity
        if counters > traffic_intensity:
            p0 = 1 / (sum((traffic_intensity ** k / math.factorial(k)) for k in range(counters)) +
                      (traffic_intensity ** counters / math.factorial(counters)) * (
                              counters / (counters - traffic_intensity)))
            pw = ((traffic_intensity ** counters / math.factorial(counters)) * (
                    counters / (counters - traffic_intensity)) * p0)

            estimated_wait_time = (pw * people_ahead) / (service_rate * counters * (1 - traffic_intensity))
        else:
            estimated_wait_time = people_ahead / total_service_capacity  # Fallback to a simple model if unstable

    return estimated_wait_time

results = []
squared_errors = []
mae_errors = []

try:
    with open('Dataset/queue_theory.txt', 'r') as file:
        for line in file:
            if line.strip():
                try:
                    data = list(map(float, line.strip().split(',')))
                    arrival_rate, service_rate, people_ahead, counters, actual_time = data
                    people_ahead = int(people_ahead)
                    predicted_time = calculate_wait_time(arrival_rate, service_rate, people_ahead, counters)
                    results.append((arrival_rate, service_rate, people_ahead, counters, predicted_time, actual_time))
                    squared_errors.append((predicted_time - actual_time) ** 2)
                    mae_errors.append(abs(predicted_time - actual_time))
                except ValueError as e:
                    print(f"Error processing line '{line.strip()}': {e}")
except FileNotFoundError:
    print("Error: The file 'queue_theory.txt' does not exist.")

mse = sum(squared_errors) / len(squared_errors) if squared_errors else 0
rmse = np.sqrt(mse)
mae = sum(mae_errors) / len(mae_errors) if mae_errors else 0
r_squared = r2_score([result[5] for result in results], [result[4] for result in results])

for result in results:
    print(
        f"Arrival Rate: {result[0]}, Service Rate: {result[1]}, People Ahead: {result[2]}, Counters: {result[3]}, Predicted Wait Time: {result[4]:.2f} seconds, Actual Wait Time: {result[5]:.2f} seconds")

print(f"\nMSE: {mse:.2f} seconds^2")
print("RMSE:", rmse)
print(f"MAE: {mae:.2f} seconds")
print(f"R-squared: {r_squared:.2f}")
