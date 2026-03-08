import json
import subprocess
from pathlib import Path

from config import JAVA_ANALYZER_JAR, DEFAULT_JAVA_FILE, FEATURES_DIR


def run_java_analyzer(java_file: Path) -> Path:
    """
    Запускает JavaProgramAnalyzer и сохраняет JSON в файл.
    Возвращает путь к JSON.
    """
    output_path = FEATURES_DIR / (java_file.stem + "_analysis.json")

    cmd = [
        "java",
        "-jar",
        str(JAVA_ANALYZER_JAR),
        str(java_file)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    output_path.write_text(result.stdout, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    json_path = run_java_analyzer(DEFAULT_JAVA_FILE)
    print(f"Analysis JSON saved to: {json_path}")
