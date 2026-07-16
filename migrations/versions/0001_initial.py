"""Initial schema — all 10 vision AI tables.

Revision ID: 0001
Revises:
Create Date: 2026-07-15
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "vision_configurations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("robot_id", sa.String(36), nullable=False, unique=True, index=True),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("thresholds", sa.JSON(), nullable=False),
        sa.Column("scene_description_mode", sa.String(50), nullable=False, server_default="RULE_BASED"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "camera_sources",
        sa.Column("source_id", sa.String(36), primary_key=True),
        sa.Column("robot_id", sa.String(36), nullable=False, index=True),
        sa.Column("source_type", sa.String(30), nullable=False),
        sa.Column("source_url", sa.String(512), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("resolution", sa.String(20), nullable=True),
        sa.Column("target_fps", sa.Integer(), nullable=False, server_default="20"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "face_profiles",
        sa.Column("face_profile_id", sa.String(36), primary_key=True),
        sa.Column("robot_id", sa.String(36), nullable=False, index=True),
        sa.Column("user_id", sa.String(36), nullable=False, index=True),
        sa.Column("display_name", sa.String(100), nullable=False),
        sa.Column("relationship", sa.String(30), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="ACTIVE"),
        sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("qdrant_point_id", sa.String(36), nullable=True),
        sa.Column("minio_prefix", sa.String(512), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "gesture_profiles",
        sa.Column("gesture_profile_id", sa.String(36), primary_key=True),
        sa.Column("robot_id", sa.String(36), nullable=False, index=True),
        sa.Column("user_id", sa.String(36), nullable=False, index=True),
        sa.Column("gesture_name", sa.String(60), nullable=False),
        sa.Column("description", sa.String(255), nullable=True),
        sa.Column("gesture_type", sa.String(20), nullable=False, server_default="STATIC"),
        sa.Column("action_hint", sa.String(60), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="ACTIVE"),
        sa.Column("is_predefined", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("qdrant_point_ids", sa.String(1024), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "vision_events",
        sa.Column("event_id", sa.String(36), primary_key=True),
        sa.Column("robot_id", sa.String(36), nullable=False, index=True),
        sa.Column("event_type", sa.String(80), nullable=False, index=True),
        sa.Column("severity", sa.String(20), nullable=False, server_default="INFO"),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, index=True),
    )

    op.create_table(
        "incident_media",
        sa.Column("media_id", sa.String(36), primary_key=True),
        sa.Column("robot_id", sa.String(36), nullable=False, index=True),
        sa.Column("event_type", sa.String(80), nullable=False),
        sa.Column("event_id", sa.String(36), nullable=True),
        sa.Column("media_type", sa.String(20), nullable=False, server_default="IMAGE"),
        sa.Column("minio_key", sa.String(512), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("captured_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, index=True),
    )

    op.create_table(
        "vision_datasets",
        sa.Column("dataset_id", sa.String(36), primary_key=True),
        sa.Column("robot_id", sa.String(36), nullable=False, index=True),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("description", sa.String(512), nullable=True),
        sa.Column("dataset_type", sa.String(40), nullable=False),
        sa.Column("image_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("label_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("minio_prefix", sa.String(512), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "training_jobs",
        sa.Column("job_id", sa.String(36), primary_key=True),
        sa.Column("robot_id", sa.String(36), nullable=False, index=True),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("training_type", sa.String(40), nullable=False),
        sa.Column("dataset_id", sa.String(36), nullable=True),
        sa.Column("base_model", sa.String(60), nullable=True),
        sa.Column("epochs", sa.Integer(), nullable=True),
        sa.Column("image_size", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="QUEUED"),
        sa.Column("logs", sa.String(8192), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("artifact_path", sa.String(512), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "model_versions",
        sa.Column("version_id", sa.String(36), primary_key=True),
        sa.Column("model_key", sa.String(40), nullable=False, index=True),
        sa.Column("version_tag", sa.String(40), nullable=False),
        sa.Column("minio_path", sa.String(512), nullable=False),
        sa.Column("training_job_id", sa.String(36), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("evaluation_id", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("activated_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "model_evaluations",
        sa.Column("eval_id", sa.String(36), primary_key=True),
        sa.Column("version_id", sa.String(36), nullable=False, index=True),
        sa.Column("model_key", sa.String(40), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("evaluated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    for table in [
        "model_evaluations", "model_versions", "training_jobs", "vision_datasets",
        "incident_media", "vision_events", "gesture_profiles", "face_profiles",
        "camera_sources", "vision_configurations",
    ]:
        op.drop_table(table)
