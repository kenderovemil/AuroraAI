import unittest
from aurora.core import Aurora
from aurora.planner import Planner
from aurora.memory import Memory

class TestAuroraCore(unittest.TestCase):
    def test_process_pipeline(self):
        a = Aurora()
        mem = Memory()
        p = Planner()
        a.attach_memory(mem)
        a.attach_planner(p)
        out = a.process('hello world')
        self.assertIn('Processed: hello world', out)
        self.assertIn('Plan: EchoPlan(hello world)', out)
        self.assertEqual(mem.retrieve('last_input'), 'hello world')

if __name__ == '__main__':
    unittest.main()

