import unittest
from unittest.mock import patch, MagicMock


class TestAgentCreation(unittest.TestCase):
    """Tests that the agent graph entry point works with the new multi-agent system."""

    @patch('agent.get_llm')
    def test_agent_creation(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_get_llm.return_value = mock_llm

        from agent import create_agent_graph
        supervisor = create_agent_graph()

        self.assertIsNotNone(supervisor)
        self.assertEqual(supervisor.name, "supervisor")

    @patch('agent.get_llm')
    def test_backward_compat_invoke(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_get_llm.return_value = mock_llm

        from agent import create_agent_graph
        supervisor = create_agent_graph()

        self.assertTrue(hasattr(supervisor, 'invoke'))
        self.assertTrue(callable(supervisor.invoke))

    @patch('agent.get_llm')
    def test_supervisor_has_all_agents(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_get_llm.return_value = mock_llm

        from agent import create_agent_graph
        supervisor = create_agent_graph()

        self.assertTrue(hasattr(supervisor, 'document_agent'))
        self.assertTrue(hasattr(supervisor, 'research_agent'))
        self.assertTrue(hasattr(supervisor, 'data_agent'))

    @patch('agent.get_llm')
    def test_get_all_traces(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_get_llm.return_value = mock_llm

        from agent import create_agent_graph
        supervisor = create_agent_graph()

        traces = supervisor.get_all_traces()
        self.assertIsInstance(traces, dict)
        self.assertIn("supervisor", traces)


if __name__ == '__main__':
    unittest.main()
