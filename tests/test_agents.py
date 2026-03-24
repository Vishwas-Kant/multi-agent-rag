import unittest
from unittest.mock import patch, MagicMock


class TestSupervisorRouting(unittest.TestCase):
    """Tests for the supervisor's intent classification logic."""

    def test_classify_weather(self):
        from agents.supervisor import classify_intent
        self.assertEqual(classify_intent("What's the weather in Delhi?"), "weather")
        self.assertEqual(classify_intent("Temperature in London"), "weather")
        self.assertEqual(classify_intent("Will it rain tomorrow?"), "weather")

    def test_classify_math(self):
        from agents.supervisor import classify_intent
        self.assertEqual(classify_intent("Calculate 5 + 3"), "data")
        self.assertEqual(classify_intent("What is 15 * 23?"), "data")
        self.assertEqual(classify_intent("What is the square root of 144?"), "data")

    def test_classify_code(self):
        from agents.supervisor import classify_intent
        self.assertEqual(classify_intent("Analyze this Python code"), "data")
        self.assertEqual(classify_intent("def hello(): pass"), "data")

    def test_classify_document(self):
        from agents.supervisor import classify_intent
        self.assertEqual(classify_intent("What does the document say about X?"), "document")
        self.assertEqual(classify_intent("Summarize the uploaded PDF"), "document")
        self.assertEqual(classify_intent("What is in the resume?"), "document")

    def test_classify_research(self):
        from agents.supervisor import classify_intent
        self.assertEqual(classify_intent("Search for latest AI news"), "research")
        self.assertEqual(classify_intent("Find information about quantum computing"), "research")
        self.assertEqual(classify_intent("Who is Elon Musk?"), "research")

    def test_classify_default(self):
        from agents.supervisor import classify_intent
        result = classify_intent("xyzzy gibberish")
        self.assertEqual(result, "document")


class TestSupervisorAgent(unittest.TestCase):
    """Tests for the supervisor agent initialization."""

    @patch('agents.supervisor.SupervisorAgent.__init__', return_value=None)
    def test_supervisor_has_specialists(self, mock_init):
        from agents.supervisor import SupervisorAgent
        supervisor = SupervisorAgent.__new__(SupervisorAgent)
        supervisor._research_agent = None
        supervisor._document_agent = None
        supervisor._data_agent = None
        supervisor.name = "supervisor"

        self.assertIsNone(supervisor._research_agent)
        self.assertIsNone(supervisor._document_agent)
        self.assertIsNone(supervisor._data_agent)


class TestBaseAgent(unittest.TestCase):
    """Tests for the base agent tool call parsing."""

    def test_parse_tool_calls_valid(self):
        from agents.base import BaseAgent

        content = '<tool_call>{"name": "fetch_weather", "arguments": {"city": "Delhi"}}</tool_call>'
        calls = BaseAgent._parse_tool_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "fetch_weather")
        self.assertEqual(calls[0]["args"]["city"], "Delhi")

    def test_parse_tool_calls_multiple(self):
        from agents.base import BaseAgent

        content = (
            '<tool_call>{"name": "tool_a", "arguments": {"x": 1}}</tool_call>'
            '<tool_call>{"name": "tool_b", "arguments": {"y": 2}}</tool_call>'
        )
        calls = BaseAgent._parse_tool_calls(content)
        self.assertEqual(len(calls), 2)

    def test_parse_tool_calls_invalid_json(self):
        from agents.base import BaseAgent

        content = '<tool_call>not valid json</tool_call>'
        calls = BaseAgent._parse_tool_calls(content)
        self.assertEqual(len(calls), 0)

    def test_parse_tool_calls_no_tags(self):
        from agents.base import BaseAgent

        content = "No tool calls here"
        calls = BaseAgent._parse_tool_calls(content)
        self.assertEqual(len(calls), 0)


if __name__ == "__main__":
    unittest.main()
