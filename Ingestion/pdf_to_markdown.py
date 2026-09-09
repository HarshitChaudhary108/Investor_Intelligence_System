import pymupdf4llm
from pathlib import Path
class PDF2Markdown:
    "convert pdf to markdown"
    def convert_pdf(self, pdf_file: str, directory: str) -> str:
        """
        convert a pdf file to markdown to preserve the 
        overall structure of the document, e.g: tables.

        Args:
            pdf_file: Source pdf file
            directory: the folder where markdown will be stored
        
        Return:
           Generate markdown file  
        """
        pdf_file = Path(pdf_file)
        
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF File do not exist")

        directory_path = Path(directory)
        if not directory_path.exists():
            directory_path.mkdir(exist_ok=True)

        markdown_file_path = directory_path/f"{pdf_file.stem}.md"

        markdown_content = pymupdf4llm.to_markdown(pdf_file)

        markdown_file_path.write_text(
            markdown_content,
            encoding="utf-8"
        )

        return str(markdown_file_path)

    def convert_directory(self, input_dir: str, output_dir: str):
        """
        Convert all the pdf's in the directory to markdown

        Args:
            input_dir: path where raw pdfs are stored
            output_dir: path where markdown file will be stored
        
        Return:
            List of generated markdown files
        """
        inputdir = Path(input_dir)

        markdown_files = []
        
        for pdf_file in inputdir.glob("*.pdf"):
            markdown_file = self.convert_pdf(
                pdf_file=pdf_file,
                directory=output_dir
            )
            markdown_files.append(markdown_file)

        return markdown_files

if __name__ == "__main__":
    obj = PDF2Markdown()
    input_dir = "data/raw_pdf"
    output_dir = "data/markdown"
    output = obj.convert_directory(
        input_dir= input_dir,
        output_dir= output_dir
        )
    print(output)