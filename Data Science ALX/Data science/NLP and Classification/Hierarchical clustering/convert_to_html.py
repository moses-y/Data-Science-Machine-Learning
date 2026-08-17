import nbformat
from nbconvert import HTMLExporter

# Specify the path to your notebook
notebook_path = "C:/Users/moses_y/OneDrive/Desktop/ML Projects/alx/Data science/NLP and Classification/clustering_guide.ipynb"
html_path = notebook_path.replace(".ipynb", ".html")

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
    notebook_content = nbformat.read(notebook_file, as_version=4)

# Convert the notebook to HTML
exporter = HTMLExporter()
html_content, _ = exporter.from_notebook_node(notebook_content)

# Save the HTML
with open(html_path, 'w', encoding='utf-8') as html_file:
    html_file.write(html_content)

print(f"Converted {notebook_path} to {html_path}")
