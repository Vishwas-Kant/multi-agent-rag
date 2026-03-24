import unittest
from unittest.mock import patch, MagicMock
from tools.weather import fetch_weather
from tools.rag import retrieve_context, initialize_rag
from tools.calculator import calculate
from tools.code_analysis import analyze_code
from tools.web_search import web_search
from tools.web_reader import read_webpage


class TestWeatherTool(unittest.TestCase):

    @patch('tools.weather._session.get')
    @patch('tools.weather.os.getenv')
    def test_fetch_weather_success(self, mock_getenv, mock_get):
        mock_getenv.return_value = "yucgxf36teg29d72ghcxiu2gwcxui"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "weather": [{"description": "sunny"}],
            "main": {"temp": 25, "feels_like": 23, "humidity": 50},
            "wind": {"speed": 5},
            "visibility": 10000,
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        from tools.weather import _fetch_weather_cached
        if hasattr(_fetch_weather_cached, 'cache'):
            _fetch_weather_cached.cache.clear()

        result = fetch_weather.invoke("London")
        self.assertIn("Weather in London", result)
        self.assertIn("sunny", result)

    @patch('tools.weather.os.getenv')
    def test_fetch_weather_no_api_key(self, mock_getenv):
        mock_getenv.return_value = None

        from tools.weather import _fetch_weather_cached
        if hasattr(_fetch_weather_cached, 'cache'):
            _fetch_weather_cached.cache.clear()

        result = fetch_weather.invoke("London")
        self.assertIn("Error: OpenWeatherMap API key not found", result)


class TestRAGTool(unittest.TestCase):

    @patch('tools.rag.PyPDFLoader')
    @patch('tools.rag.RecursiveCharacterTextSplitter')
    @patch('tools.rag.get_embeddings')
    @patch('tools.rag.get_vector_store')
    @patch('tools.rag.save_vector_store')
    @patch('tools.rag.os.path.exists')
    def test_initialize_rag(self, mock_exists, mock_save, mock_get_vs, mock_get_emb, mock_splitter, mock_loader):
        mock_exists.return_value = True
        mock_loader_instance = mock_loader.return_value
        mock_loader_instance.load.return_value = [MagicMock(page_content="test content", metadata={})]

        mock_splitter_instance = mock_splitter.return_value
        mock_splitter_instance.split_documents.return_value = [MagicMock(page_content="test", metadata={"extraction_method": "pypdf"})]

        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock()
        mock_get_vs.return_value = None

        with patch('tools.rag.FAISS') as mock_faiss:
            mock_faiss.from_documents.return_value = mock_vs
            result = initialize_rag(["dummy.pdf"])
            self.assertIn("RAG system initialized", result)

    def test_retrieve_context_not_initialized(self):
        import tools.rag
        tools.rag._retriever = None

        # Clear cache
        from tools.rag import retrieve_context_cached
        if hasattr(retrieve_context_cached, 'cache'):
            retrieve_context_cached.cache.clear()

        result = retrieve_context.invoke("query")
        self.assertIn("Error: RAG system not initialized", result)


class TestCalculatorTool(unittest.TestCase):

    def test_basic_arithmetic(self):
        result = calculate.invoke("2 + 3")
        self.assertIn("= 5", result)

    def test_multiplication(self):
        result = calculate.invoke("15 * 23")
        self.assertIn("= 345", result)

    def test_sqrt(self):
        result = calculate.invoke("sqrt(144)")
        self.assertIn("= 12", result)

    def test_trigonometry(self):
        result = calculate.invoke("sin(radians(30))")
        self.assertIn("0.5", result)

    def test_pi_constant(self):
        result = calculate.invoke("pi")
        self.assertIn("3.14159", result)

    def test_complex_expression(self):
        result = calculate.invoke("2 ** 10")
        self.assertIn("= 1024", result)

    def test_division_by_zero(self):
        result = calculate.invoke("1 / 0")
        self.assertIn("Error", result)

    def test_invalid_expression(self):
        result = calculate.invoke("import os")
        self.assertIn("Error", result)

    def test_empty_expression(self):
        result = calculate.invoke("")
        self.assertIn("Error", result)


class TestCodeAnalysisTool(unittest.TestCase):

    def test_analyze_function(self):
        code = '''
            def greet(name: str) -> str:
                """Greet someone."""
                return f"Hello, {name}!"
        '''
        result = analyze_code.invoke(code)
        self.assertIn("greet", result)
        self.assertIn("Functions", result)

    def test_analyze_class(self):
        code = '''
            class Dog:
                """A dog class."""
                def bark(self):
                    return "Woof!"
                def fetch(self, item):
                    return f"Fetched {item}"
        '''
        result = analyze_code.invoke(code)
        self.assertIn("Dog", result)
        self.assertIn("Classes", result)
        self.assertIn("bark", result)

    def test_analyze_imports(self):
        code = 'import os\nfrom pathlib import Path'
        result = analyze_code.invoke(code)
        self.assertIn("Imports", result)

    def test_syntax_error(self):
        code = "def broken(:"
        result = analyze_code.invoke(code)
        self.assertIn("Syntax error", result)

    def test_empty_code(self):
        result = analyze_code.invoke("")
        self.assertIn("Error", result)


class TestWebSearchTool(unittest.TestCase):

    @patch('tools.web_search.DDGS', create=True)
    def test_web_search_success(self, MockDDGS=None):
        try:
            from duckduckgo_search import DDGS
            result = web_search.invoke({"query": "test", "max_results": 1})
            self.assertIsInstance(result, str)
        except ImportError:
            result = web_search.invoke({"query": "test", "max_results": 1})
            self.assertIn("Error", result)

    def test_web_search_empty_query(self):
        result = web_search.invoke({"query": "", "max_results": 1})
        self.assertIsInstance(result, str)


class TestWebReaderTool(unittest.TestCase):

    def test_read_webpage_invalid_url(self):
        result = read_webpage.invoke("https://en.wikipedia.org/wiki/India#History")
        self.assertIsInstance(result, str)


if __name__ == '__main__':
    unittest.main()
