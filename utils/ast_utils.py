import tree_sitter_python as tspython
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser
 
PY_LANGUAGE = Language(tspython.language(), "python")
JAVA_LANGUAGE = Language(tsjava.language(), "java")
 
def get_ast(code, lang):
    parser = Parser()
    if lang == "python": parser.set_language(PY_LANGUAGE)
    elif lang == "java": parser.set_language(JAVA_LANGUAGE)
    return parser.parse(bytes(code, "utf-8")).root_node
