import os
import csv
import re

def process_log_files(folder):
    csv_file = f"{folder}.csv"
    data = []

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            file_path = os.path.join(folder, file)
            with open(file_path, 'r') as f:
                content = f.read()

                if not content:
                    print(f"Warning: {file} is empty. Skipping.")
                    continue

                instance = file.split('_')[1]  # Extract instance number from filename
                obj = re.search(r'TT=(\d+)', content)
                obj = obj.group(1) if obj else None
                status = re.search(r'status=(\w+)', content)
                status = status.group(1) if status else None
                gap = re.search(r'gap=([\d.]+)%', content)
                gap = float(gap.group(1)) if gap else None
                time = re.search(r'time=([\d.]+)s', content)
                time = float(time.group(1))/3600 if time else None  # Convert seconds to hours

                if any([obj, status, gap, time]):
                    data.append([instance, obj or "N/A", status or "N/A", f"{gap:.2f}" if gap else "N/A", f"{time:.2f}" if time else "N/A"])

    data.sort(key=lambda x: int(x[0]))  # Sort by instance number

    with open(csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Instance', 'Obj', 'Status', 'Gap (%)', 'Time (h)'])
        csvwriter.writerows(data)

    return csv_file

def csv_to_latex(csv_file):
    latex_table = f"""
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{|c|c|c|c|c|}}
\\hline
Instance & Obj & Status & Gap (\%) & Time (h) \\\\
\\hline
"""

    with open(csv_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header
        for row in csvreader:
            latex_table += ' & '.join(row) + ' \\\\\n'

    latex_table += f"""\\hline
\\end{{tabular}}
\\caption{{Résultats pour {os.path.basename(csv_file).split('.')[0]}}}
\\end{{table}}
"""

    return latex_table

def main():
    folders = ['out/20j_5m', 'out/50j_10m']

    for folder in folders:
        csv_file = process_log_files(folder)
        latex_table = csv_to_latex(csv_file)
        print(latex_table)
        
        with open(f"{os.path.basename(folder)}_table.tex", 'w') as f:
            f.write(latex_table)

if __name__ == "__main__":
    main()