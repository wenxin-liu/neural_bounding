from pathlib import Path
from typing import Dict, Any


class DataExporter:
    def __init__(self, object_name, dimension, query, filename="result"):
        self.directory_path: Path = Path()
        self.csv_file = None
        self.filename: str = filename
        self.save_results: Dict[int, Dict[str, Any]] = {}

        self._create_directory(object_name, dimension, query)

    def _create_directory(self, object_name, dimension, query) -> None:
        """
        Create a new directory for storing results within the 'results' directory.
        """
        project_root = Path(__file__).resolve().parents[2]
        results_dir = project_root / 'results' / query / f'{dimension}D' / object_name
        results_dir.mkdir(parents=True, exist_ok=True)
        self.directory_path = results_dir

    def _append_line_to_csv(self, filename: str, line: str) -> None:
        if self.csv_file is None:
            self.csv_file = open(f"{self.directory_path}/{filename}", "w")

        self.csv_file.write(f"{line}\n")

    def export_results(self, metrics_registry) -> None:
        header = "method,class weights,iteration,false negatives,false positives,true values,total samples,loss"
        self._append_line_to_csv(filename=f'{self.filename}.csv', line=header)

        for method_key, method_values in metrics_registry.metrics_registry.items():
            line = f'{method_key},'+','.join(map(str, method_values.values()))
            self._append_line_to_csv(filename=f'{self.filename}.csv', line=line)

        self.csv_file.close()
