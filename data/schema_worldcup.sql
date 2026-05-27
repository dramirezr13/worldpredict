-- Tablas para datos del paquete worldcup-master (Fjelstul World Cup Database)

CREATE TABLE IF NOT EXISTS worldcup_tournaments (
    tournament_id TEXT PRIMARY KEY,
    tournament_name TEXT,
    year INTEGER,
    start_date DATE,
    end_date DATE,
    host_country TEXT,
    winner TEXT,
    count_teams INTEGER
);

CREATE TABLE IF NOT EXISTS worldcup_teams (
    team_id TEXT PRIMARY KEY,
    team_name TEXT NOT NULL,
    team_code TEXT,
    confederation_code TEXT,
    region_name TEXT
);

CREATE TABLE IF NOT EXISTS worldcup_qualified_teams (
    id SERIAL PRIMARY KEY,
    tournament_id TEXT NOT NULL REFERENCES worldcup_tournaments(tournament_id),
    team_id TEXT,
    team_name TEXT NOT NULL,
    team_code TEXT,
    group_name TEXT,
    confederation TEXT,
    is_host BOOLEAN DEFAULT FALSE,
    performance TEXT,
    UNIQUE (tournament_id, team_name)
);

CREATE INDEX IF NOT EXISTS idx_wc_qualified_tournament
    ON worldcup_qualified_teams (tournament_id);

CREATE INDEX IF NOT EXISTS idx_wc_qualified_group
    ON worldcup_qualified_teams (tournament_id, group_name);

CREATE TABLE IF NOT EXISTS worldcup_matches (
    match_id TEXT PRIMARY KEY,
    tournament_id TEXT NOT NULL,
    tournament_year INTEGER,
    match_date DATE,
    stage_name TEXT,
    group_name TEXT,
    home_team_name TEXT NOT NULL,
    away_team_name TEXT NOT NULL,
    home_team_score INTEGER,
    away_team_score INTEGER,
    knockout_stage BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_wc_matches_tournament
    ON worldcup_matches (tournament_id);
