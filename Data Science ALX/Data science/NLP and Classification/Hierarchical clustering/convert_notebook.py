import nbformat
from nbconvert import PythonExporter

# Specify the path to your notebook
notebook_path = "C:/Users/moses_y/OneDrive/Desktop/ML Projects/alx/Data science/NLP and Classification/clustering_guide.ipynb"
script_path = notebook_path.replace(".ipynb", ".py")

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
    notebook_content = nbformat.read(notebook_file, as_version=4)

# Convert the notebook to Python script
exporter = PythonExporter()
script_content, _ = exporter.from_notebook_node(notebook_content)

# Save the script
with open(script_path, 'w', encoding='utf-8') as script_file:
    script_file.write(script_content)

print(f"Converted {notebook_path} to {script_path}")
