import importlib
import sys
import types
import unittest
from unittest.mock import patch


class AnalysisImportTests(unittest.TestCase):
    def test_import_does_not_require_pipeline_type_at_transformers_top_level(self) -> None:
        module_name = "tp2_imagenes.analysis"
        previous_module = sys.modules.pop(module_name, None)
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.pipeline = lambda **_kwargs: None

        try:
            with patch.dict(sys.modules, {"transformers": fake_transformers}):
                analysis = importlib.import_module(module_name)
            self.assertEqual(analysis.MODEL_ID, "PekingU/rtdetr_r50vd")
        finally:
            sys.modules.pop(module_name, None)
            if previous_module is not None:
                sys.modules[module_name] = previous_module


if __name__ == "__main__":
    unittest.main()
