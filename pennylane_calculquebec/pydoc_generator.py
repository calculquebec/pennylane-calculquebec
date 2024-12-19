"""
Contains an experiment function for generating documentation using pydoc\n
this is a WIP and should probably not be used   
"""

import os
import pydoc

def generate_pydoc(directory, output_dir):
    """
    Generates pydoc documentation for all Python modules in the specified directory and its subdirectories.
    
    Args:   
        directory (str): The root directory to search for Python files.
        output_dir (str): The directory to save the generated HTML documentation.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                # Get the module path
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)
                module_name = relative_path.replace(os.sep, ".").replace(".py", "")
                
                try:
                    # Generate the HTML documentation
                    print(f"Generating documentation for module: {module_name}")
                    output_file = os.path.join(output_dir, f"{module_name}.html")
                    with open(output_file, "w") as f:
                        f.write(pydoc.HTMLDoc().docmodule(pydoc.importfile(file_path)))
                except Exception as e:
                    print(f"Failed to generate documentation for {module_name}: {e}")

# Example usage
if __name__ == "__main__":
    source_dir = "."  # Replace with the directory containing your Python files
    output_dir = "./doc_output"        # Replace with the desired output directory
    generate_pydoc(source_dir, output_dir)