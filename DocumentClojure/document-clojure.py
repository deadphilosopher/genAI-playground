"""
Clojure Code Documentation Generator

Usage:
  document_clojure.py <input_dir> <output_dir>
  document_clojure.py (-h | --help)

Arguments:
  <input_dir>   Path to the directory containing Clojure (.clj) files.
  <output_dir>  Path to the directory where the documentation will be saved.

Options:
  -h --help     Show this help message.
"""

import os
from docopt import docopt
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

def main(input_dir, output_dir):
    # Initialize the Ollama LLM with the required model
    llm = OllamaLLM(model="codestral:22b")  # Replace "llama-2" with your specific model name

    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=["file_name", "file_content"],
        template=(
            "You are a programming assistant proficient in Clojure. "
            "Your task is to review the following Clojure files and create detailed documentation."
            "File Name:"
            "{file_name}"
            "Code:"
            "{file_content}"
            "Documentation:"
            "- High-level summary of the file's purpose."
            "- Detailed description of each function (parameters, return values, and examples)."
            "- Areas where comments or clarity could be improved."
        )
    )

    # Create the RunnableSequence chain
    chain = prompt | llm

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process all files in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".clj"):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, file.replace(".clj", ".txt"))

                # Read the Clojure file
                with open(input_file_path, "r") as f:
                    file_content = f.read()

                # Generate documentation
                documentation = chain.invoke({"file_name": file, "file_content": file_content})

                # Write the documentation to the output directory
                with open(output_file_path, "w") as f:
                    f.write(documentation)
                print(f"Documentation created for {file} -> {output_file_path}")


if __name__ == "__main__":
    # Parse arguments using docopt
    args = docopt(__doc__)
    input_dir = args["<input_dir>"]
    output_dir = args["<output_dir>"]

    # Run the main function
    main(input_dir, output_dir)
