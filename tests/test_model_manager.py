import pytest
from app.ai.model_manager import ModelManager, MODEL_KEYS


class TestModelManager:
    def setup_method(self):
        self.manager = ModelManager()

    def test_initial_status_has_all_model_keys(self):
        statuses = self.manager.get_status()
        keys_returned = [s["model_key"] for s in statuses]
        for key in MODEL_KEYS:
            assert key in keys_returned

    def test_get_status_for_specific_key(self):
        statuses = self.manager.get_status("object-detector")
        assert len(statuses) == 1
        assert statuses[0]["model_key"] == "object-detector"

    def test_get_status_for_unknown_key_returns_empty(self):
        statuses = self.manager.get_status("does-not-exist")
        assert statuses == []

    def test_is_ready_returns_false_before_load(self):
        assert not self.manager.is_ready("object-detector")

    def test_load_logic_only_model_marks_ready(self):
        # gesture-recognizer is logic-only — no actual model to load
        result = self.manager.load_model("gesture-recognizer")
        assert result is True
        assert self.manager.is_ready("gesture-recognizer")

    def test_set_enabled_false_for_logic_only_model(self):
        ok, msg = self.manager.set_enabled("gesture-recognizer", False)
        assert ok
        assert not self.manager._status["gesture-recognizer"]["enabled"]

    def test_set_enabled_unknown_key_returns_false(self):
        ok, msg = self.manager.set_enabled("unknown-model", True)
        assert not ok
        assert "Unknown model key" in msg

    def test_dependency_check_blocks_disabling_face_detector(self):
        """Cannot disable face-detector when face-recognizer is enabled."""
        self.manager._status["face-recognizer"]["enabled"] = True
        ok, msg = self.manager.set_enabled("face-detector", False)
        assert not ok
        assert "face-recognizer" in msg

    def test_unload_marks_model_as_disabled(self):
        self.manager.load_model("gesture-recognizer")
        self.manager.unload_model("gesture-recognizer")
        assert not self.manager._status["gesture-recognizer"]["loaded"]
