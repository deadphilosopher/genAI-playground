import subprocess
import os
from typing import Optional
from tree_sitter import Language, Parser
from langchain.tools import BaseTool
from pydantic import PrivateAttr


class CppParserTool(BaseTool):
    name: str = "cpp_parser"
    description: str = "Parse C++ code to extract its syntax tree or analyze its structure using Tree-Sitter."
    language_lib_path: str  # Explicitly define the field with a type annotation

    # Define private attributes for non-Pydantic fields
    _parser: Parser = PrivateAttr()
    _cpp_language: Language = PrivateAttr()


    def __init__(self, language_lib_path: str):
        # Initialize the Pydantic fields
        super().__init__(language_lib_path=language_lib_path)

        # Load the compiled language library
        self._cpp_language = Language.load(language_lib_path)
        self._parser = Parser()
        self._parser.set_language(self._cpp_language)


    def _run(self, code: str) -> str:
        """
        Parse the input C++ code and return its syntax tree.
        """
        if not code.strip():
            raise ValueError("Empty code provided. Please input valid C++ code.")

        tree = self._parser.parse(code.encode("utf8"))
        return self._print_tree(tree.root_node, code)


    def _print_tree(self, node, code, indent=0) -> str:
        """
        Recursively print the syntax tree.
        """
        lines = []
        node_text = code[node.start_byte:node.end_byte]
        lines.append(f"{'  ' * indent}{node.type}: '{node_text}'")
        for child in node.children:
            lines.extend(self._print_tree(child, code, indent + 1))
        return "\n".join(lines)


def build_tree_sitter_library():
    import subprocess
    import os

    repo_url = "https://github.com/tree-sitter/tree-sitter-cpp.git"
    clone_dir = "./tree-sitter-cpp"
    lib_name = "tree-sitter-cpp.so"

    # Clone the repository
    if not os.path.exists(clone_dir):
        subprocess.run(["git", "clone", repo_url, clone_dir], check=True)

    # Build the shared library manually
    parser_file = os.path.join(clone_dir, "src", "parser.c")
    scanner_file = os.path.join(clone_dir, "src", "scanner.c")  # Update if needed
    output_file = lib_name

    # Compile for amd64 (x86_64) architecture
    subprocess.run(
        [
            "gcc",
            "-shared",
            "-o", output_file,
            "-fPIC",
            "-arch", "x86_64",  # Specify the amd64 target
            parser_file,
            scanner_file,  # Include the scanner file
        ],
        check=True
    )
    print(f"Tree-Sitter library built at {output_file} for amd64")


if __name__ == "__main__":
    # Build the library
    build_tree_sitter_library()

    # Initialize the C++ parser tool
    cpp_tool = CppParserTool(language_lib_path="./tree-sitter-cpp.so")

    # Example usage
    example_code = """
    int main() {
        std::cout << "Hello, World!" << std::endl;
        return 0;
    }
    """
    syntax_tree = cpp_tool.run(example_code)
    print(syntax_tree)
