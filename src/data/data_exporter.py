from pathlib import Path
from typing import Dict, Any


class DataExporter:
    def __init__(self, directory_name: str):
        self.directory_path: Path = Path()
        self.csv_file = None
        self.filename: str = ""
        self.save_results: Dict[int, Dict[str, Any]] = {}

        self._create_directory(directory_name)

    def _create_directory(self, directory_name: str) -> None:
        """
        Create a new directory for storing results within the 'exporter_data' directory.

        Parameters:
            directory_name (str): The name of the directory to be created.
        """

        project_root = Path(__file__).resolve().parents[2]
        results_dir = project_root / 'exporter_data' / directory_name

        results_dir.mkdir(parents=True, exist_ok=True)
        self.directory_path = results_dir

    def _append_line_to_csv(self, filename: str, line: str) -> None:
        if self.csv_file is None:
            self.csv_file = open(f"{self.directory_path}/{filename}", "w")
            self.filename = filename

        self.csv_file.write(f"{line}\n")

    def save_experiment_results(self, class_weight: float, metrics_registry: Any, iteration: int, loss: float = 0.0) -> None:
        metrics = metrics_registry.get_metrics()

        self.save_results[iteration] = {
            "class weight": class_weight,
            "iteration": iteration,
            "false negatives": metrics["false_negative"],
            "false positives": metrics["false_positive"],
            "true values": metrics["true_value"],
            "total samples": metrics["total_samples"],
            "loss": f"{loss:.5f}"
        }

    def export_results(self) -> None:
        header = "class weights,iteration,false negatives,false positives,true values,total samples,loss"
        self._append_line_to_csv(filename="result.csv", line=header)

        for iteration_results in self.save_results.values():
            line = ','.join(map(str, iteration_results.values()))
            self._append_line_to_csv(filename="result.csv", line=line)

        if self.csv_file:
            self.csv_file.close()