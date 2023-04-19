from typing import Optional, Union
import dataclasses
import textwrap

import logging
import ast

import pylsp
from pylsp import hookimpl, uris
import openai

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.root.setLevel(logging.DEBUG)

REFACTOR_DOCSTRING_COMMAND = "refactor.gpt.docstring"


@hookimpl
def pylsp_commands(config, workspace):
    api_key = config.plugin_settings("pylsp_gpt").get("openai-api-key")
    if api_key is None:
        logger.warn("No OpenAI API key provided. Disabling pylsp_gpt plugin.")
        return []

    openai.api_key = api_key
    logger.info("pylsp_gpt enabled.")
    return [
        REFACTOR_DOCSTRING_COMMAND,
    ]


@hookimpl
def pylsp_execute_command(config, workspace, command, arguments):
    """Execute a LSP command in the given workspace based on the
    provided configuration and command line arguments. If the command
    is a refactor docstring command, the GPTDocstringCommand is
    executed on the specified document and range, and the resulting
    workspace edit is applied to the workspace.

    """
    logger.info("workspace/executeCommand: %s %s", command, arguments)

    api_key = config.plugin_settings("pylsp_gpt").get("openai-api-key")
    if api_key is None:
        logger.warn("No OpenAI API key provided. Disabling pylsp_gpt plugin.")
        return []

    openai.api_key = api_key
    
    logger.info("pylsp_gpt enabled.")
    if command == REFACTOR_DOCSTRING_COMMAND:
        document_uri, range = arguments
        document = workspace.get_document(document_uri)

        refactor = GPTDocstringCommand(document, range)

        workspace_edit = refactor.execute()
        logger.info("Applying workspace edit: %s", workspace_edit)
        workspace.apply_edit(workspace_edit)


DocstringableNode = Union[
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module
]


class GPTDocstringCommand:
    def __init__(
        self, document: pylsp.workspace.Document, range: dict[str, dict[str, int]]
    ):
        self.document = document
        self.range = range
        # Parse the AST of the document.
        self.document_module = ast.parse(self.document.source, self.document.path)
        self.node = self.find_enclosing_ast_node()

    def execute(self):
        # Find the enclosing AST node for self.range within self.document
        # Generate a new docstring for the node
        new_docstring = self.generate_docstring_text()
        new_docstring += " " * self.calculate_docstring_indentation_level()

        # Compute whether this is a replacement or an insertion.
        current_docstring = _get_current_docstring_node(self.node)
        if current_docstring is not None:
            # Replace the current docstring.
            range = {
                "start": {
                    "line": current_docstring.lineno - 1,
                    "character": current_docstring.col_offset,
                },
                "end": {
                    "line": current_docstring.end_lineno - 1,
                    "character": current_docstring.end_col_offset,
                },
            }
        else:
            # Insert a new docstring.
            indentation = self.calculate_docstring_indentation_level()
            range = {
                "start": {"line": self.node.lineno, "character": indentation},
                "end": {"line": self.node.lineno, "character": indentation},
            }

        workspace_edit = {
            "changes": {
                self.document.uri: [
                    {
                        "range": range,
                        "newText": new_docstring,
                    }
                ]
            }
        }
        return workspace_edit

    def generate_docstring_prompt(self) -> str:
        """
        Generates a new docstring for the given node by querying ChatGPT.

        Returns:
            A string that represents a template prompt for a new docstring.
            The prompt includes the type and name of the given node, and provides
            guidelines for writing a docstring using the Google style for docstrings.
    
        Raises:
            TypeError: If the type of the given node is not recognized.
        """
        
        # Generate the prompt.

        if isinstance(self.node, ast.FunctionDef):
            node_type = "function"
            node_name = self.node.name
        elif isinstance(self.node, ast.AsyncFunctionDef):
            node_type = "async function"
            node_name = self.node.name
        elif isinstance(self.node, ast.ClassDef):
            node_type = "class"
            node_name = self.node.name
        elif isinstance(self.node, ast.Module):
            node_type = "module"
            node_name = self.document.path
        else:
            raise TypeError(f"Unknown node type: {type(self.node)}")

        prompt = f"""
Please write new docstring for the following {node_type} (named '{self.node.name}').
ONLY WRITE A DOCSTRING FOR THE {node_type.upper()} AND NOTHING ELSE.


Use the Google style for docstrings.
        
For very short functions, you may omit the Args and Returns sections.
For very short functions, you may write a single-line docstring.

Wrap all text at 80 characters.

Here is the {node_type} you are documenting:

{self.extract_enclosing_ast_node_source()}
        """
        print(prompt)
        return prompt

    def generate_docstring_text(self) -> str:
        """Generate a new docstring for the given node by querying ChatGPT."""
        # Generate the prompt.
        prompt = self.generate_docstring_prompt()
        # Send the query to ChatGPT.
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a large language model generating documentation strings for Python code.
                    RESPOND WITH THE DOCSTRING AND ONLY WITH THE DOCSTRING.
                    Do not include the Python triple quotes in your response.
                    """,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        docstring = response.choices[0].message.content
        return self.sanitize_docstring_text(docstring)

    def sanitize_docstring_text(self, docstring_text: str) -> str:
        """
        Sanitize the given docstring text by removing the triple
        quotes at the beginning and end. Then add a newline to the
        beginning and end of the text. Ensure that the docstring is
        indented correctly and add the triple quotes back to the
        string before returning.

        """
        
        # Remove the triple quotes; we'll add them back later.
        if docstring_text.startswith('"""') or docstring_text.startswith("'''"):
            docstring_text = docstring_text[3:]
        if docstring_text.endswith('"""') or docstring_text.endswith("'''"):
            docstring_text = docstring_text[:-3]

        # Add a newline to the beginning and end.
        if not docstring_text.startswith("\n"):
            docstring_text = "\n" + docstring_text
        if not docstring_text.endswith("\n"):
            docstring_text += "\n"

        # Ensure that the docstring is indented correctly.
        indentation = self.calculate_docstring_indentation_level()
        docstring_text = textwrap.indent(docstring_text, " " * indentation)

        # Add the triple quotes.
        docstring_text = '"""' + docstring_text + '"""\n'

        return docstring_text

    def extract_enclosing_ast_node_source(self) -> str:
        """Extract the source code for the given node."""
        # Python's ast package is 1-indexed for line numbers.
        return "".join(self.document.lines[self.node.lineno - 1 : self.node.end_lineno])

    def calculate_docstring_indentation_level(self) -> int:
        # Use the indendation level of the first expression/statement inside the node.
        for child in self.node.body:
            if isinstance(child, (ast.expr, ast.stmt)):
                return child.col_offset
        # Fall back to the indentation level of the parent.
        return self.node.col_offset

    def find_enclosing_ast_node(self) -> DocstringableNode:
        """Analyze the module and determine the target location for
        the docstring as well as the bounding node. The function uses
        Python's abstract syntax tree (AST) to determine the
        tightest-enclosing node of a given region. If a node is not
        found, the function returns the top-level module. The return
        value is a `DocstringableNode`, which is a class that
        represents a node in the AST that can have a docstring
        attached to it.

        """
        
        # Find the tightest-enclosing node of the given region
        start_line, start_col = (
            self.range["start"]["line"],
            self.range["start"]["character"],
        )
        end_line, end_col = self.range["end"]["line"], self.range["end"]["character"]

        # Python's ast package is 1-indexed for line numbers.
        start_line += 1
        end_line += 1

        def dfs_walk(node, stack):
            if isinstance(node, (ast.expr, ast.stmt)):
                if node.lineno <= start_line and node.end_lineno >= end_line:
                    # The node encloses the given region
                    stack.append(node)
                    for child in ast.iter_child_nodes(node):
                        dfs_walk(child, stack)

        matching_stack = []
        for node in self.document_module.body:
            stack = []
            dfs_walk(node, stack)
            if len(stack) > 0:
                matching_stack = stack
                break

        # Walk back up the tree until we find a node that is a function, class or module
        matching_node = None
        while len(matching_stack) > 0:
            node = matching_stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return node

        # No matching node found, so we're at the module level.
        return self.document_module


def _get_current_docstring_node(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module]
) -> Optional[ast.Expr]:
    """Get the existing docstring AST node for the given node."""
    if len(node.body) == 0:
        return None
    first = node.body[0]
    if not isinstance(first, ast.Expr):
        return None

    first_val = first.value
    if isinstance(first_val, ast.Str):
        return first
    if isinstance(first_val, ast.Constant) and isinsatnce(first_val.value, str):
        return first

    return None


@hookimpl
def pylsp_definitions(config, workspace, document, position):
    logger.info("textDocument/definition: %s %s", document, position)

    
    filename = __file__
    uri = uris.uri_with(document.uri, path=filename)
    with open(filename) as f:
        lines = f.readlines()
        for lineno, line in enumerate(lines):
            if "def pylsp_definitions" in line:
                break
    return [
        {
            "uri": uri,
            "range": {
                "start": {
                    "line": lineno,
                    "character": 4,
                },
                "end": {
                    "line": lineno,
                    "character": line.find(")") + 1,
                },
            },
        }
    ]
