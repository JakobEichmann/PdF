import os
from pathlib import Path

# Корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Пути
JAVA_ANALYZER_JAR = PROJECT_ROOT / "java-analyzer" / "target" / "java-analyzer-1.0-SNAPSHOT-jar-with-dependencies.jar"
TESTCASES_DIR = PROJECT_ROOT / "testcases"
FEATURES_DIR = PROJECT_ROOT / "python-pipeline" / "features"

# Создавать директорию для фич
os.makedirs(FEATURES_DIR, exist_ok=True)

# Модель для CodeBERT
CODEBERT_MODEL_NAME = "microsoft/codebert-base"

# Имя тестового файла по умолчанию
DEFAULT_JAVA_FILE = TESTCASES_DIR / "IntervalExample.java"
# --------------------------
# Phi-3 settings (LLM)
# --------------------------
PHI3_MODEL_NAME = os.getenv("PHI3_MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
PHI3_MODEL_PATH = os.getenv("PHI3_MODEL_PATH", "")  # если пусто, используется PHI3_MODEL_NAME

PHI3_MAX_NEW_TOKENS = int(os.getenv("PHI3_MAX_NEW_TOKENS", "512"))
PHI3_DO_SAMPLE = os.getenv("PHI3_DO_SAMPLE", "0") == "1"
PHI3_TEMPERATURE = float(os.getenv("PHI3_TEMPERATURE", "0.0"))
PHI3_TOP_P = float(os.getenv("PHI3_TOP_P", "1.0"))
PHI3_REPETITION_PENALTY = float(os.getenv("PHI3_REPETITION_PENALTY", "1.05"))