import subprocess
import os

def latex_to_pdf(latex_file_path):
    # Ensure the file exists
    if not os.path.exists(latex_file_path):
        raise FileNotFoundError(f"The file {latex_file_path} does not exist.")

    # Define the output directory
    output_directory = 'C:\\Users\\ketan\\Desktop\\SPAIDER-SPACE\\sagan_workflow\\output'

    # Run pdflatex command with output directory
    try:
        subprocess.run(['pdflatex', '-output-directory', output_directory, latex_file_path], check=True)
        print("PDF generated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

# Example usage
# latex_to_pdf('C:\\Users\\ketan\\Desktop\\SPAIDER-SPACE\\sagan_workflow\\output\\output.tex')