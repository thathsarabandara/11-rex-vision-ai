import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from app.ai.scene_understanding import SceneUnderstanding
from app.ai.scene_captioner import SceneCaptioner
from app.pipelines.scene_pipeline import run_scene_pipeline


class TestSceneUnderstanding:
    def setup_method(self):
        self.su = SceneUnderstanding()

    def test_empty_scene(self):
        scene = self.su.build([], [], [], [])
        assert scene["person_count"] == 0
        assert scene["pet_count"] == 0
        assert scene["detected_gestures"] == []

    def test_counts_persons_correctly(self):
        detections = [
            {"class_name": "person", "confidence": 0.9, "track_id": 1},
            {"class_name": "person", "confidence": 0.8, "track_id": 2},
            {"class_name": "dog", "confidence": 0.7, "track_id": 3},
        ]
        scene = self.su.build(detections, [], [], [])
        assert scene["person_count"] == 2
        assert scene["pet_count"] == 1

    def test_known_faces_extracted(self):
        faces = [
            {"status": "KNOWN", "display_name": "Thathsara"},
            {"status": "UNKNOWN", "display_name": None},
        ]
        scene = self.su.build([], faces, [], [])
        assert "Thathsara" in scene["known_faces"]
        assert scene["unknown_face_count"] == 1

    def test_security_observation_set_for_unknown(self):
        faces = [{"status": "UNKNOWN", "display_name": None}]
        scene = self.su.build([], faces, [], [])
        assert "unknown person" in scene["security_observation"]

    def test_low_light_flag_propagated(self):
        scene = self.su.build([], [], [], [], low_light=True)
        assert scene["low_light"] is True

    def test_important_objects_exclude_people_and_pets(self):
        detections = [
            {"class_name": "person"}, {"class_name": "laptop"},
            {"class_name": "cup"}, {"class_name": "cat"},
        ]
        scene = self.su.build(detections, [], [], [])
        assert "person" not in scene["important_objects"]
        assert "cat" not in scene["important_objects"]
        assert "laptop" in scene["important_objects"]

    def test_gesture_names_collected(self):
        gestures = [{"gesture_name": "THUMBS_UP"}, {"gesture_name": "OPEN_PALM"}]
        scene = self.su.build([], [], gestures, [])
        assert "THUMBS_UP" in scene["detected_gestures"]


class TestSceneCaptioner:
    def setup_method(self):
        self.captioner = SceneCaptioner()

    def test_empty_scene_description(self):
        scene = {"person_count": 0, "pet_count": 0, "known_faces": [],
                 "unknown_face_count": 0, "important_objects": [],
                 "activities": [], "low_light": False, "security_observation": ""}
        desc = self.captioner.describe(scene)
        assert "No people" in desc

    def test_single_known_person_description(self):
        scene = {"person_count": 1, "pet_count": 0, "known_faces": ["Thathsara"],
                 "unknown_face_count": 0, "important_objects": [],
                 "activities": [], "low_light": False, "security_observation": ""}
        desc = self.captioner.describe(scene)
        assert "Thathsara" in desc

    def test_low_light_mentioned(self):
        scene = {"person_count": 0, "pet_count": 0, "known_faces": [],
                 "unknown_face_count": 0, "important_objects": [],
                 "activities": [], "low_light": True, "security_observation": ""}
        desc = self.captioner.describe(scene)
        assert "Low-light" in desc

    def test_multiple_people_description(self):
        scene = {"person_count": 3, "pet_count": 0, "known_faces": ["Alice"],
                 "unknown_face_count": 2, "important_objects": ["laptop"],
                 "activities": [], "low_light": False, "security_observation": ""}
        desc = self.captioner.describe(scene)
        assert "3 people" in desc

    def test_security_observation_included(self):
        scene = {"person_count": 1, "pet_count": 0, "known_faces": [],
                 "unknown_face_count": 1, "important_objects": [],
                 "activities": [], "low_light": False, "security_observation": "unknown person present"}
        desc = self.captioner.describe(scene)
        assert "Security observation" in desc


@pytest.mark.asyncio
async def test_scene_pipeline_returns_tuple():
    result = await run_scene_pipeline(
        detections=[{"class_name": "person", "confidence": 0.9}],
        face_results=[{"status": "UNKNOWN"}],
        gesture_results=[],
        movement_results=[],
        low_light=False,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    scene, description = result
    assert isinstance(scene, dict)
    assert isinstance(description, str)
