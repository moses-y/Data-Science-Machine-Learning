
import os
import requests
import nbformat
import joblib
import pandas as pd
import numpy as np
import markdown
from io import StringIO
import matplotlib.pyplot as plt

import torch
import webbrowser
import jinja2
from transformers import GPT2Tokenizer, GPT2LMHeadModel


from typing import List, Optional, Dict, Union, Any
from pydantic import BaseModel, validator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


def scan_and_process_files(directory, extensions, process_file=None):
    """
    Scans the directory recursively for files with given extensions and optionally processes them.
    
    :param directory: The directory to search in.
    :param extensions: A set or list of file extensions to include.
    :param process_file: Optional; A function to call with each found file's path.
    :return: A tuple containing two elements:
        - A dictionary with file extensions as keys and counts as values.
        - A list of file paths that were found.
    """
    file_counts = {ext: 0 for ext in extensions}
    file_paths = []

    for root, _, files in os.walk(directory):
        matched_files = [file for file in files if any(file.endswith(ext) for ext in extensions)]
        file_paths.extend(os.path.join(root, file) for file in matched_files)
        for file in matched_files:
            ext = next((ext for ext in extensions if file.endswith(ext)), None)
            if ext:
                file_counts[ext] += 1
                if process_file:
                    process_file(os.path.join(root, file))

    return file_counts, file_paths

# Example process_file function
def process_example(file_path):
    print(f"Processing file: {file_path}")

# Example usage
directory = r"C:\Users\moses_y\OneDrive\Desktop\ML Projects\alx\Data science\Machine Learning\Random Forests"
extensions = ['.py', '.ipynb']

file_counts, found_files = scan_and_process_files(directory, extensions, process_example)

print("Found files:")
for file in found_files:
    print(file)

print("\nFile counts:")
for ext, count in file_counts.items():
    print(f"{ext}: {count}")


class GPT2Summarizer:
    def __init__(self, model_name='gpt2', model_checkpoint_path=None):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side='left')
        if model_checkpoint_path:
            self.model = GPT2LMHeadModel.from_pretrained(model_name, state_dict=torch.load(model_checkpoint_path))
        else:
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to inference mode
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Ensure tokenizer's pad_token is set to eos_token

    def generate_summary(self, text: str, summary_length: int = 1024, length_penalty: float = 2.0, temperature: float = 0.5, top_p: float = 0.75, num_beams: int = 6) -> str:
        inputs = self.tokenizer.encode_plus(text, return_tensors="pt", max_length=720, truncation=True, padding="max_length")
        summary_ids = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=summary_length,
            min_length=40,
            length_penalty=length_penalty,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    def summarize_markdown_and_notebooks(self, markdown_path: str, notebooks_paths: list) -> dict:
        summaries = {}
        try:
            print(f"Opening Markdown file at: {markdown_path}")  # Log the path
            with open(markdown_path, 'r', encoding='utf-8') as md_file:
                markdown_text = md_file.read()
            summaries['markdown_report'] = self.generate_summary(markdown_text)

            for nb_path in notebooks_paths:
                print(f"Opening Notebook file at: {nb_path}")  # Log each path
                with open(nb_path, 'r', encoding='utf-8') as nb_file:
                    nb = nbformat.read(nb_file, as_version=4)
                    # Adjusted line to ensure non-empty markdown and code cells are summarized
                    nb_text = "".join(cell['source'] for cell in nb['cells'] if cell['cell_type'] in ['markdown', 'code'] and cell['source'].strip())
                summaries[nb_path] = self.generate_summary(nb_text)
        except Exception as e:
            print(f"Error during summarization: {e}")
        return summaries

class RFParams(BaseModel):
    n_estimators: List[int]
    max_depth: List[Optional[int]] = None
    min_samples_split: List[Optional[int]]

    @validator('max_depth', 'min_samples_split', pre=True, each_item=True)
    def check_none(cls, v):
        return v if v is not None else [None]

class ModelPerformance(BaseModel):
    RMSE: float
    R2: float
    model_params: Dict[str, Union[int, None]]
    feature_importances: Dict[str, float]  # Adjusted to match the provided key

class RandomForestPipeline:
    def __init__(self, dataset_url: str, target_column: str, params: RFParams, model_checkpoint_path):
        self.dataset_url = dataset_url
        self.target_column = target_column
        self.params = params.dict(exclude_none=True)  # Exclude None to avoid grid search errors
        self.models_performance: List[ModelPerformance] = []
        self.feature_importances = []
        self.model_checkpoint_path = model_checkpoint_path
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side='left')
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", state_dict=torch.load(model_checkpoint_path))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.summaries = []

    def load_data(self):
        try:
            df = pd.read_csv(self.dataset_url)
            df = pd.get_dummies(df, drop_first=True)
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            return train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            print(f"Failed to load data: {e}")
            raise

    def train_and_evaluate(self):
        X_train, X_test, y_train, y_test = self.load_data()
        grid_search = GridSearchCV(RandomForestRegressor(random_state=42), 
                                   self.params, 
                                   cv=5, 
                                   scoring='neg_mean_squared_error', 
                                   verbose=1)
        grid_search.fit(X_train, y_train)

        best_estimator = grid_search.best_estimator_
        predictions = best_estimator.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        feature_importances = {feature: importance for feature, importance in zip(X_train.columns, best_estimator.feature_importances_)}

        self.models_performance.append(ModelPerformance(
            RMSE=rmse,
            R2=r2,
            model_params=grid_search.best_params_,
            feature_importances=feature_importances
        ))

        # Store metrics and parameters
        metrics = {
            "RMSE": rmse,
            "R2": r2,
            "model_params": grid_search.best_params_,
            "feature_importances": feature_importances
        }
        self.models_performance.append(metrics)

    def save_metrics(self, filename="metrics.json"):
        import json
        with open(filename, "w") as f:
            json.dump(self.models_performance, f)

    def save_model(self, model_performance: ModelPerformance, model_filename: str = "random_forest_model.joblib"):
        # Locate and save the best estimator from model_performance
        best_estimator = next((model for model in self.models_performance if model == model_performance), None)
        if best_estimator:
            joblib.dump(best_estimator, model_filename)

    def calculate_averages(self):
        average_rmse = np.mean([model.RMSE for model in self.models_performance])
        average_r2 = np.mean([model.R2 for model in self.models_performance])
        return {"average_rmse": average_rmse, "average_r2": average_r2}
        
    def visualize_feature_importances(self) -> List[str]:
        """
        Generates a plot for the feature importances of the last trained model.
        Returns the path to the saved plot.
        """
        if not self.models_performance:
            print("No models trained yet.")
            return ""
        
        plot_paths = []
        for idx, perf in enumerate(self.models_performance, start=1):
            features = list(perf.feature_importances.keys())
            importances = list(perf.feature_importances.values())
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(10, 6))
            plt.title(f"Feature Importances for Model {idx}")
            plt.barh(range(len(indices)), [importances[i] for i in indices], color='b', align='center')
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            plot_path = f"feature_importances_{idx}.png"
            plt.savefig(plot_path)
            plt.close()
            plot_paths.append(plot_path)
        
        return plot_paths        

class ReportGenerator:
    def __init__(self, models_performance: List[Dict[str, Any]], summaries: Dict[str, str], template_path: str):
        self.models_performance = models_performance
        self.summaries = summaries
        self.template_path = template_path

    def generate_markdown_report(self, output_file_path: str):
        content = "# Model Performance Report\n\n"
        for perf in self.models_performance:
            content += f"## Model: {perf['model_name']}\n"
            content += f"- RMSE: {perf['RMSE']}\n"
            content += f"- R^2: {perf['R2']}\n\n"
            content += "### Feature Importances\n"
            for feature, importance in perf['feature_importances'].items():
                content += f"- {feature}: {importance}\n"
            content += "\n---\n"
        # Optionally include GPT-2 generated summaries
        for title, summary in self.summaries.items():
            content += f"## Summary: {title}\n{summary}\n\n"
        with open(output_file_path, "w", encoding='utf-8') as file:  # Set encoding to 'utf-8'
            file.write(content)

    def generate_html_report(self, output_file_path: str, project_name: str, average_rmse: float, average_r2: float, feature_importance_paths: List[str]):
        markdown_content = self.generate_markdown_content()
        html_content = markdown.markdown(markdown_content)
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(self.template_path))
        template = env.get_template("report_template.html")
        context = {
            "project_name": project_name,
            "average_rmse": average_rmse,
            "average_r2": average_r2,
            "models_performance": self.models_performance,
            "summaries": self.summaries,
            "feature_importance_paths": feature_importance_paths,
        }
        rendered_html = template.render(context)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(rendered_html)
        print(f"HTML report generated: {output_file_path}")

    def generate_markdown_content(self) -> str:
        content = "# Model Performance Report\n\n"
        for perf in self.models_performance:
            content += f"## Model: {perf['model_name']}\n"
            content += f"- RMSE: {perf['RMSE']}\n"
            content += f"- R^2: {perf['R2']}\n\n"
            content += "### Feature Importances\n"
            for feature, importance in perf['feature_importances'].items():
                content += f"- {feature}: {importance}\n"
            content += "\n---\n"
        return content

if __name__ == "__main__":
    # Define model parameters
    params = RFParams(
        n_estimators=[20, 50, 100, 200],
        max_depth=[20, 15, 10, 5],
        min_samples_split=[6, 5, 4, 3]  # Updated to use valid values
    )
    # Path to your model checkpoint
    model_checkpoint_path = "C:\\Users\\moses_y\\OneDrive\\Desktop\\ML Projects\\Transformers\\src\\models\\model_checkpoint_epoch_3.pt"
    # Initialize the pipeline with dataset information and hyperparameters
    pipeline_manager = RandomForestPipeline(
        dataset_url="https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/Python/Crop_yield.csv", 
        target_column="Yield", 
        params=params,
        model_checkpoint_path=model_checkpoint_path
    )
    
    # Train the model and evaluate its performance
    model_performance = pipeline_manager.train_and_evaluate()
    
    # Save the model to a file
    pipeline_manager.save_model(model_performance, model_filename="random_forest_model.joblib")

    pipeline_manager.save_metrics()
 
    # Visualize feature importances
    feature_importance_paths = pipeline_manager.visualize_feature_importances()
    
    # Paths for the markdown report and the Jinja2 template, both located in the script's directory
    script_directory = os.path.dirname(os.path.realpath(__file__))
    markdown_report_path = os.path.join(script_directory, "model_performance_report.md")
    template_path = script_directory  #  The 'report_template.html' is here

    # The path to the project directory where notebooks are located
    project_path = r"C:\Users\moses_y\OneDrive\Desktop\ML Projects\alx\Data science\Machine Learning\Random Forests"
    extensions = ['.py', '.ipynb']
    
    # Fetching notebooks paths
    _, notebooks_paths = scan_and_process_files(project_path, extensions, process_example)
    
    # Filter out only .ipynb files for summarization
    notebooks_paths = [path for path in notebooks_paths if path.endswith(".ipynb")]
    print("Notebook files for summarization:", notebooks_paths)

    # Prepare data for the report generator
    models_performance = [{
        "model_name": f"Model with {perf.model_params['n_estimators']} estimators",
        "RMSE": perf.RMSE,
        "R2": perf.R2,
        "feature_importances": perf.feature_importances
    } for perf in pipeline_manager.models_performance]

    averages = pipeline_manager.calculate_averages()
    average_rmse = averages['average_rmse']
    average_r2 = averages['average_r2']    

    # Summaries for markdown report and notebooks
    summarizer = GPT2Summarizer(model_name='gpt2', model_checkpoint_path=model_checkpoint_path)
    summaries = summarizer.summarize_markdown_and_notebooks(
        markdown_path=markdown_report_path,
        notebooks_paths=notebooks_paths
    )

    # Generate reports
    report_generator = ReportGenerator(
        models_performance=models_performance,
        summaries=summaries,
        template_path=template_path
    )

    # Markdown report
    report_generator.generate_markdown_report(output_file_path="model_performance_report.md")

    # HTML report
    report_generator.generate_html_report(
        output_file_path="final_report.html",
        project_name="ALX Crop Yield Prediction",
        average_rmse=pipeline_manager.calculate_averages()['average_rmse'],
        average_r2=pipeline_manager.calculate_averages()['average_r2'],
        feature_importance_paths=feature_importance_paths
    )
