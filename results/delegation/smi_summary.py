import csv
import sys

def parse_row(row):
    timestamp, index, temperature_gpu, power_draw, clocks_sm, clocks_mem, utilization_gpu, memory_used, clocks_event_reasons_active = row
    temperature_gpu = int(temperature_gpu.strip())
    power_draw = float(power_draw.strip().split()[0])
    clocks_sm = int(clocks_sm.strip().split()[0])
    utilization_gpu = int(utilization_gpu.strip().split('%')[0])
    return index, temperature_gpu, power_draw, clocks_sm, utilization_gpu

def main(csv_path):
    data = {}
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            index, temperature_gpu, power_draw, clocks_sm, utilization_gpu = parse_row(row)
            if index not in data:
                data[index] = {
                    'max_temp': temperature_gpu,
                    'power_draws': [],
                    'min_sm_at_util50': float('inf'),
                    'rows_util50': 0
                }
            data[index]['max_temp'] = max(data[index]['max_temp'], temperature_gpu)
            data[index]['power_draws'].append(power_draw)
            if utilization_gpu >= 50:
                data[index]['min_sm_at_util50'] = min(data[index]['min_sm_at_util50'], clocks_sm)
                data[index]['rows_util50'] += 1

    for index in sorted(data.keys()):
        mean_power = round(sum(data[index]['power_draws']) / len(data[index]['power_draws']), 1)
        max_power = max(data[index]['power_draws'])
        min_sm_at_util50 = f"{data[index]['min_sm_at_util50']}MHz" if data[index]['rows_util50'] > 0 else 'N/A'
        print(f"GPU {index}: max_temp={data[index]['max_temp']}C mean_power={mean_power}W max_power={max_power:.1f}W min_sm_at_util50={min_sm_at_util50} rows_util50={data[index]['rows_util50']}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python smi_summary.py <csv_file_path>")
        sys.exit(1)
    main(sys.argv[1])
