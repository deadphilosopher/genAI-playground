"""
Code Documentation Generator

Usage:
  code_doc_generator.py <input_dir> <output_dir>
  code_doc_generator.py (-h | --help)

Arguments:
  <input_dir>   Path to the directory containing source code files.
  <output_dir>  Path to the directory where the documentation will be saved.

Options:
  -h --help     Show this help message.
"""

import os
from docopt import docopt
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

# Language to file extension mapping
LANGUAGE_EXTENSIONS = {
    "Clojure": {".clj"},
    "C++": {".cpp", ".hpp", ".h"}
}

def determine_language(file_extension):
    """Determine the programming language based on file extension."""
    for language, extensions in LANGUAGE_EXTENSIONS.items():
        if file_extension in extensions:
            return language
    return None

def get_prompt(language):
    """Get a language-specific prompt template."""
    return PromptTemplate(
        input_variables=["file_name", "file_content"],
        template=(
            f"You are a programming assistant proficient in {language}. "
            f"Your task is to review the following {language} file and create detailed documentation.\n\n"
            "File Name: {file_name}\n\n"
            "Code:\n{file_content}\n\n"
            "Documentation:\n"
            "- High-level summary of the file's purpose.\n"
            "- Detailed description of each function, class, and key elements (parameters, return values, usage examples).\n"
            "- Areas where comments or clarity could be improved."
        )
    )

def main(input_dir, output_dir):
    # Initialize the Ollama LLM with the "codestral:22b" model
    llm = OllamaLLM(model="codestral:22b")

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process all files in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_extension = os.path.splitext(file)[1]
            language = determine_language(file_extension)

            if language:
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, file.replace(file_extension, f"_{language}.txt"))

                # Read the source file
                with open(input_file_path, "r") as f:
                    file_content = f.read()

                # Get the appropriate prompt for the language
                prompt = get_prompt(language)
                chain = prompt | llm

                # Generate documentation
                documentation = chain.invoke({"file_name": file, "file_content": file_content})

                # Write the documentation to the output directory
                with open(output_file_path, "w") as f:
                    f.write(documentation)
                print(f"Documentation created for {file} ({language}) -> {output_file_path}")


if __name__ == "__main__":
    # Parse arguments using docopt
    args = docopt(__doc__)
    input_dir = args["<input_dir>"]
    output_dir = args["<output_dir>"]

    # Run the main function
    main(input_dir, output_dir)
