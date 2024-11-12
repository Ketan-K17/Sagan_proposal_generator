import markdown
import pdfkit

def convert_md_to_pdf(md_file_path, pdf_file_path):
    # Read the Markdown file
    with open(md_file_path, 'r', encoding='utf-8') as md_file:
        md_content = md_file.read()

    # Convert Markdown to HTML
    html_content = markdown.markdown(md_content)

    # Convert HTML to PDF
    pdfkit.from_string(html_content, pdf_file_path)

# Example usage
# md_file = 'C:\\Users\\ketan\\Desktop\\SPAIDER-SPACE\\sagan_workflow\\output\\output.md'
# pdf_file = 'C:\\Users\\ketan\\Desktop\\SPAIDER-SPACE\\sagan_workflow\\output\\output.pdf'
# convert_md_to_pdf(md_file, pdf_file)