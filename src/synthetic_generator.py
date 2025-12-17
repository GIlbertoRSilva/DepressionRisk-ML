import numpy as np
import pandas as pd

class SyntheticInputGenerator:
    def __init__(self, random_state=42):
        self.random_state = random_state

        self.binary_cols = [
            "Gender",
            "Have you ever had suicidal thoughts ?",
            "Family History of Mental Illness"
        ]

        self.integer_cols = [
            "Academic Pressure",
            "Study Satisfaction",
            "Sleep Duration",
            "Dietary Habits",
            "Work/Study Hours",
            "Financial Stress"
        ]

        self.continuous_cols = [
            "Age",
            "CGPA"
        ]

        self.columns_order = [
            "Gender",
            "Age",
            "Academic Pressure",
            "CGPA",
            "Study Satisfaction",
            "Sleep Duration",
            "Dietary Habits",
            "Have you ever had suicidal thoughts ?",
            "Work/Study Hours",
            "Financial Stress",
            "Family History of Mental Illness"
        ]

    def fit(self, df):
        np.random.seed(self.random_state)

        # BINÁRIAS → probabilidade de 1
        self.binary_probs = {
            col: df[col].mean()
            for col in self.binary_cols
        }

        # INTEIRAS → média e desvio
        self.integer_stats = {
            col: {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max()
            }
            for col in self.integer_cols
        }

        # CONTÍNUAS
        self.cont_stats = {
            col: {
                "mean": df[col].mean(),
                "std": df[col].std()
            }
            for col in self.continuous_cols
        }

        return self

    def sample(self, n_samples):
        data = {}

        # BINÁRIAS
        for col, p in self.binary_probs.items():
            data[col] = np.random.binomial(1, p, n_samples)

        # INTEIRAS
        for col, stats in self.integer_stats.items():
            values = np.random.normal(
                stats["mean"],
                stats["std"],
                n_samples
            )
            values = np.round(values)
            values = np.clip(values, stats["min"], stats["max"])
            data[col] = values.astype(int)

        # CONTÍNUAS
        data["Age"] = np.round(
            np.random.normal(
                self.cont_stats["Age"]["mean"],
                self.cont_stats["Age"]["std"],
                n_samples
            )
        ).astype(int)

        data["CGPA"] = np.round(
            np.random.normal(
                self.cont_stats["CGPA"]["mean"],
                self.cont_stats["CGPA"]["std"],
                n_samples
            ),
            2
        )

        df = pd.DataFrame(data)
        return df[self.columns_order]
