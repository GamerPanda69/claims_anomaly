-- Healthcare Claims Fraud Detection — Seed Data Only
-- Table DDL is now managed by SQLAlchemy (Base.metadata.create_all).
-- This file is kept only to seed the default admin user.

-- Create default superuser (password: admin123 - CHANGE IN PRODUCTION!)
INSERT INTO users (username, email, password_hash, role, is_active)
VALUES (
    'admin',
    'admin@frauddetection.local',
    '$2b$12$DvfarmHo0abG3iqd1kjLDewOkhWW0Ldgu333J.U04/IoyO6wrFVNi',
    'superuser',
    true
)
ON CONFLICT (username) DO NOTHING;
